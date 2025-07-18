'''
# author: H.
# email: 
# date: xxx

The code is mainly modified from GitHub link below:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py
'''

import os
import argparse
import logging

import math
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from torch.nn import init
from typing import Union
from utils.registry import BACKBONE

logger = logging.getLogger(__name__)


from torch import Tensor
import numpy as np
import math
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import pickle
import random


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,freq_scale='Mel'):

        super(SincConv,self).__init__()


        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels+1
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)


        if freq_scale == 'Mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.freq=filbandwidthsf[:self.out_channels]

        elif freq_scale == 'Inverse-mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.mel=filbandwidthsf[:self.out_channels]
            self.freq=np.abs(np.flip(self.mel)-1) ## invert mel scale

        
        else:
            fmelmax=np.max(f)
            fmelmin=np.min(f)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            self.freq=filbandwidthsmel[:self.out_channels]
        
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels-1,self.kernel_size)
    
       
        
    def forward(self,x):
        for i in range(len(self.freq)-1):
            fmin=self.freq[i]
            fmax=self.freq[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels-1, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


        
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)



import copy
@BACKBONE.register_module(module_name="rawnet2")
class RawNet(nn.Module):

    def __init__(self, config):
        super(RawNet, self).__init__()
        config = copy.deepcopy(config)
        self.device=config['device']

        self.Sinc_conv=SincConv(device=self.device,
			out_channels = config['filts'][0],
			kernel_size = config['first_conv'],
                        in_channels = config['in_channels'],freq_scale='Mel'
        )
        # filts: [128, [128, 128], [128, 512], [512, 512]]
        self.first_bn = nn.BatchNorm1d(num_features = config['filts'][0]) # 128
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts = config['filts'][1], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = config['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = config['filts'][2]))
        config['filts'][2][0] = config['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts = config['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = config['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = config['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features = config['filts'][1][-1],
            l_out_features = config['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features = config['filts'][1][-1],
            l_out_features = config['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features = config['filts'][2][-1],
            l_out_features = config['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features = config['filts'][2][-1],
            l_out_features = config['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features = config['filts'][2][-1],
            l_out_features = config['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features = config['filts'][2][-1],
            l_out_features = config['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features = config['filts'][2][-1])
        self.gru = nn.GRU(input_size = config['filts'][2][-1],
			hidden_size = config['gru_node'],
			num_layers = config['nb_gru_layer'],
			batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = config['gru_node'], # 1024
			out_features = config['nb_fc_node']) # 1024
       
        self.fc2_gru = nn.Linear(in_features = config['nb_fc_node'], # 1024
			out_features = config['nb_classes'],bias=True) # 2
			
       
        self.sig = nn.Sigmoid()
        
        self.init_weight()

    def init_weight(self, std=0.2):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                nn.init.uniform_(m.running_mean, 0, 0.1)
                nn.init.uniform_(m.running_var, 0, 0.1)
                nn.init.uniform_(m.weight, 0, 0.1)
                nn.init.uniform_(m.bias, 0, 0.1)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def features(self, x, y = None,inference=False):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x=x.view(nb_samp,1,len_seq)
        
        x = self.Sinc_conv(x)    # Fixed sinc filters convolution
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        
        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1) # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)
        

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1) # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1 # (batch, filter, time) x (batch, filter, 1)

        # print(f"x shape:{x.shape}, block2.bn1: {print(self.block2[0].bn1)}")
        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1) # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2 # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1) # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3 # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1) # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4 # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1) # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5 # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.fc1_gru(x)
        
        return x
        # 🥏 We Use in RawNet2! but not used in ucf-audio!
        # feat = x
        # x = self.fc2_gru(x)
        # if not inference:
        #     output = x
        #     return output, feat
        # else:
        #     output=F.softmax(x,dim=1)
        #     return output, feat
      
    def forward(self, x, y = None,inference=False):
        x = self.features(x, y, inference)
        x = self.fc2_gru(x)

        if not inference:
            output = x
            return output

        else:
            output=F.softmax(x,dim=1)
            return output
        

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []
        
        l_fc.append(nn.Linear(in_features = in_features,
			        out_features = l_out_features))

        

        return nn.Sequential(*l_fc)


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

    # def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
    #     if print_fn == None: printfn = print
    #     model = self
        
    #     def register_hook(module):
    #         def hook(module, input, output):
    #             class_name = str(module.__class__).split(".")[-1].split("'")[0]
    #             module_idx = len(summary)
                
    #             m_key = "%s-%i" % (class_name, module_idx + 1)
    #             summary[m_key] = OrderedDict()
    #             summary[m_key]["input_shape"] = list(input[0].size())
    #             summary[m_key]["input_shape"][0] = batch_size
    #             if isinstance(output, (list, tuple)):
    #                 summary[m_key]["output_shape"] = [
	# 					[-1] + list(o.size())[1:] for o in output
	# 				]
    #             else:
    #                 summary[m_key]["output_shape"] = list(output.size())
    #                 if len(summary[m_key]["output_shape"]) != 0:
    #                     summary[m_key]["output_shape"][0] = batch_size
                        
    #             params = 0
    #             if hasattr(module, "weight") and hasattr(module.weight, "size"):
    #                 params += torch.prod(torch.LongTensor(list(module.weight.size())))
    #                 summary[m_key]["trainable"] = module.weight.requires_grad
    #             if hasattr(module, "bias") and hasattr(module.bias, "size"):
    #                 params += torch.prod(torch.LongTensor(list(module.bias.size())))
    #             summary[m_key]["nb_params"] = params
                
    #         if (
	# 			not isinstance(module, nn.Sequential)
	# 			and not isinstance(module, nn.ModuleList)
	# 			and not (module == model)
	# 		):
    #             hooks.append(module.register_forward_hook(hook))
                
    #     device = device.lower()
    #     assert device in [
	# 		"cuda",
	# 		"cpu",
	# 	], "Input device is not valid, please specify 'cuda' or 'cpu'"
        
    #     if device == "cuda" and torch.cuda.is_available():
    #         dtype = torch.cuda.FloatTensor
    #     else:
    #         dtype = torch.FloatTensor
    #     if isinstance(input_size, tuple):
    #         input_size = [input_size]
    #     x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    #     summary = OrderedDict()
    #     hooks = []
    #     model.apply(register_hook)
    #     model(*x)
    #     for h in hooks:
    #         h.remove()
            
    #     print_fn("----------------------------------------------------------------")
    #     line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    #     print_fn(line_new)
    #     print_fn("================================================================")
    #     total_params = 0
    #     total_output = 0
    #     trainable_params = 0
    #     for layer in summary:
    #         # input_shape, output_shape, trainable, nb_params
    #         line_new = "{:>20}  {:>25} {:>15}".format(
	# 			layer,
	# 			str(summary[layer]["output_shape"]),
	# 			"{0:,}".format(summary[layer]["nb_params"]),
	# 		)
    #         total_params += summary[layer]["nb_params"]
    #         total_output += np.prod(summary[layer]["output_shape"])
    #         if "trainable" in summary[layer]:
    #             if summary[layer]["trainable"] == True:
    #                 trainable_params += summary[layer]["nb_params"]
    #         print_fn(line_new)
