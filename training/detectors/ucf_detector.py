'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the UCFDetector

# author: Hainan ren
# email: hnren666@gmail.com
# date: 2023-1101
# description: Class for the UCFDetector in Audio!

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@article{yan2023ucf,
  title={UCF: Uncovering Common Features for Generalizable Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
  journal={arXiv preprint arXiv:2304.13949},
  year={2023}
}

!!
'''

import os
import datetime
import logging
import random
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

try:
    from .dim_losses import fenchel_dual_loss, infonce_loss, donsker_varadhan_loss
except:
    from dim_losses import fenchel_dual_loss, infonce_loss, donsker_varadhan_loss

logger = logging.getLogger(__name__)

# 🥏 MI loss ..

def compute_dim_loss(l_enc, m_enc, measure, mode):
    '''Computes DIM loss.

    Args:
        l_enc: Local feature map encoding.
        m_enc: Multiple globals feature map encoding.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.

    Returns:
        torch.Tensor: Loss.

    '''

    if mode == 'fd':
        loss = fenchel_dual_loss(l_enc, m_enc, measure=measure)
    elif mode == 'nce':
        loss = infonce_loss(l_enc, m_enc)
    elif mode == 'dv':
        loss = donsker_varadhan_loss(l_enc, m_enc)
    else:
        raise NotImplementedError(mode)

    return loss


@DETECTOR.register_module(module_name='ucf')
class UCFDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['backbone_config']['num_classes']
        self.encoder_feat_dim = config['encoder_feat_dim']
        
        # 📢📢📢📢 TO TRAIN WITH MI, WE HAD BETTER ALIGN THESE FEATURES!!
        # self.half_fingerprint_dim = self.encoder_feat_dim//2
        self.half_fingerprint_dim = self.encoder_feat_dim
        self.encoder_f = self.build_backbone(config) # the `forgery` feature should be furthur decoupled!
        self.encoder_c = self.build_backbone(config) # the `content` we do not use to do prediction!

        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # conditional gan
        self.con_gan = Conditional_UNet_Audio()

        # head
        specific_task_number = 7 # 7 for LibriSeVoc
        self.head_spe = Head(
            in_f=self.half_fingerprint_dim, 
            hidden_dim=self.encoder_feat_dim,
            out_f=specific_task_number
        )
        self.head_sha = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.block_spe = Conv2d1x1( # specific forgery!
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim, 
            out_f=self.half_fingerprint_dim
        )
        self.block_sha = Conv2d1x1( # common forgery!
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.half_fingerprint_dim, 
            out_f=self.half_fingerprint_dim
        )
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        if eval(config['pretrained']):
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self, config):
        cls_loss_class = LOSSFUNC[config['loss_func']['cls_loss']]
        spe_loss_class = LOSSFUNC[config['loss_func']['spe_loss']]
        con_loss_class = LOSSFUNC[config['loss_func']['con_loss']]
        rec_loss_class = LOSSFUNC[config['loss_func']['rec_loss']]
        cls_loss_func = cls_loss_class()
        spe_loss_func = spe_loss_class()
        con_loss_func = con_loss_class(margin=3.0)
        rec_loss_func = rec_loss_class()
        loss_func = {
            'cls': cls_loss_func, 
            'spe': spe_loss_func,
            'con': con_loss_func,
            'rec': rec_loss_func,
        }
        return loss_func
    
    def features(self, data_dict: dict, inference=False) -> torch.tensor:
        cat_data = data_dict['audio']
        # encoder
        f_all = self.encoder_f.features(cat_data, inference = inference)
        c_all = self.encoder_c.features(cat_data, inference = inference)
        feat_dict = {'forgery': f_all, 'content': c_all}
        return feat_dict

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # classification, multi-task
        # split the features into the specific and common forgery
        features = torch.unsqueeze(features, dim=2)
        features = torch.unsqueeze(features, dim=3)
        f_spe = self.block_spe(features)
        f_share = self.block_sha(features)
        return f_spe, f_share
    
    def get_losses(self, data_dict: dict, pred_dict: dict, inference=False) -> dict:
        if 'label_spe' in data_dict and 'reconstruction_audios' in pred_dict and not inference:
            return self.get_train_losses(data_dict, pred_dict)
        else:  # test mode
            return self.get_test_losses(data_dict, pred_dict)

    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get combined, real, fake imgs
        cat_data = data_dict['audio']
        real_audio, fake_audio = cat_data.chunk(2, dim=0)
        # get the reconstruction imgs
        reconstruction_audio_1, \
        reconstruction_audio_2, \
        self_reconstruction_audio_1, \
        self_reconstruction_audio_2 \
            = pred_dict['reconstruction_audios']
        # get label
        label = data_dict['label']
        label_spe = data_dict['label_spe']
        # get pred
        pred = pred_dict['cls']
        pred_spe = pred_dict['cls_spe']

        # 1. classification loss for common features
        loss_sha = self.loss_func['cls'](pred, label) # predict the Real or Fake
        
        # 2. classification loss for specific features
        loss_spe = self.loss_func['spe'](pred_spe, label_spe) # predict the real, GAN1, GAN2

        # 3. reconstruction loss
        # print(f'fake_audio: {fake_audio.shape}, self_reconstruction_audio_1: {self_reconstruction_audio_1.shape}')
        self_loss_reconstruction_1 = self.loss_func['rec'](fake_audio, self_reconstruction_audio_1)
        self_loss_reconstruction_2 = self.loss_func['rec'](real_audio, self_reconstruction_audio_2)
        cross_loss_reconstruction_1 = self.loss_func['rec'](fake_audio, reconstruction_audio_2)
        cross_loss_reconstruction_2 = self.loss_func['rec'](real_audio, reconstruction_audio_1)
        loss_reconstruction = \
            self_loss_reconstruction_1 + self_loss_reconstruction_2 + \
            cross_loss_reconstruction_1 + cross_loss_reconstruction_2

        # 4. constrative loss
        common_features = pred_dict['feat']
        specific_features = pred_dict['feat_spe']
        loss_con = self.loss_func['con'](common_features, specific_features, label_spe)

        # 🥏 MI loss ..
        # apply the MI in the 'content' features and the 'common' features.
        # Goal:  To make the content and common be more similar, make the common feature be more general.
        # content:  pred_dict['feat_content']
        # common:   pred_dict['feat']
        # specific: pred_dict['feat_spe']

        if self.config['mi_param']['mi_turn_on']:
            feat_content = pred_dict['feat_content']
            feat_common = pred_dict['feat']
            feat_specific = pred_dict['feat_spe']
            
            b, u  = feat_content.shape # batch, num_units
            feat_content = feat_content.reshape(b, u, -1)
            feat_common = feat_common.reshape(b, u, -1)
            feat_specific = feat_specific.reshape(b, u, -1)
            # f-gan DV loss: requires [B, Units, others(M*M) ] shape!
            # We Had better reshape to this!
            # loss_mi = compute_dim_loss(feat_content, feat_common, measure = 'JSD', mode='fd')
            loss_mi = compute_dim_loss(feat_content, feat_common, measure = self.config['mi_param']['mi_mode_measure'], mode=self.config['mi_param']['mi_mode'])
        
        
        # 5. total loss
        loss = loss_sha + 0.1*loss_spe + 0.3*loss_reconstruction + 0.05*loss_con
        if self.config['mi_param']['mi_turn_on']:
            loss += float(self.config['mi_param']['mi_lambda']) * loss_mi

        loss_dict = {
            'overall': loss,
            'common': loss_sha,
            'specific': loss_spe,
            'reconstruction': loss_reconstruction,
            'contrastive': loss_con,
        }
        if self.config['mi_param']['mi_turn_on']:
            loss_dict['mi'] = loss_mi
        return loss_dict

    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get label
        label = data_dict['label']
        # get pred
        pred = pred_dict['cls']
        # for test mode, only classification loss for common features
        loss = self.loss_func['cls'](pred, label)
        loss_dict = {'common': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy
        
        # get pred and label
        label = data_dict['label']
        pred = pred_dict['cls']
        label_spe = data_dict['label_spe']
        pred_spe = pred_dict['cls_spe']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'acc_spe': acc_spe, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        # import pdb;pdb.set_trace()
        if self.prob and self.label:
            y_pred = np.concatenate(self.prob)
            y_true = np.concatenate(self.label)
            # auc
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            # eer
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            # ap
            ap = metrics.average_precision_score(y_true,y_pred)
            # acc
            acc = self.correct / self.total
            # reset the prob and label
            self.prob, self.label = [], []
            del y_pred
            del y_true
        else:
            acc, auc, eer, ap = 0., 0., 0., 0.

        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':None, 'label':None}

    def visualize_features(self, specific_features, common_features):
        import matplotlib.pyplot as plt

        # Assuming that features are 1D tensors
        specific_features = specific_features.detach().numpy()[0]
        common_features = common_features.detach().numpy()[0]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        axes[0].bar(range(len(specific_features)), specific_features)
        axes[0].set_title('Specific Features')

        axes[1].bar(range(len(common_features)), common_features)
        axes[1].set_title('Common Features')

        plt.savefig('features.png')

    def forward(self, data_dict: dict, inference=False) -> dict:
        # split the features into the content and forgery
        features = self.features(data_dict, inference = inference)
        forgery_features, content_features = features['forgery'], features['content']
        # get the prediction by classifier (split the common and specific forgery)
        f_spe, f_share = self.classifier(forgery_features)

        if inference:
            # inference only consider share loss
            out_sha, sha_feat = self.head_sha(f_share)
            out_spe, spe_feat = self.head_spe(f_spe)
            prob_sha = torch.softmax(out_sha, dim=1)[:, 1]
            self.prob.append(
                prob_sha
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(out_sha, 1)
            correct = (prediction_class == data_dict['label']).sum().detach().cpu().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

            # pred_dict = {'cls': out_sha.detach().cpu(), 'feat': sha_feat.detach().cpu()}
            pred_dict = {'cls': out_sha.detach(), 'feat': sha_feat.detach(), 'feat_content': content_features.detach(), 'feat_specific': spe_feat.detach()}
            return  pred_dict

        bs = self.config['train_batchSize']
        # using idx aug in the training mode
        aug_idx = random.random()
        if aug_idx < 0.7:
            # real
            idx_list = list(range(0, bs//2))
            random.shuffle(idx_list)
            f_share[0: bs//2] = f_share[idx_list]
            # fake
            idx_list = list(range(bs//2, bs))
            random.shuffle(idx_list)
            f_share[bs//2: bs] = f_share[idx_list]
        
        # concat spe and share to obtain new_f_all
        f_all = torch.cat((f_spe, f_share), dim=1)
        
        # reconstruction loss
        f2, f1 = f_all.chunk(2, dim=0) # the f1 or f2 its dim1 contain the specific-forgery and common-forgery features!
        c2, c1 = content_features.chunk(2, dim=0)

        # ==== self reconstruction ==== #
        # f1 + c1 -> f11, f11 + c1 -> near~I1
        self_reconstruction_audio_1 = self.con_gan(f1, c1)

        # f2 + c2 -> f2, f2 + c2 -> near~I2
        self_reconstruction_audio_2 = self.con_gan(f2, c2)

        # ==== cross combine ==== #
        reconstruction_audio_1 = self.con_gan(f1, c2)
        reconstruction_audio_2 = self.con_gan(f2, c1)

        # head for spe and sha
        out_spe, spe_feat = self.head_spe(f_spe)
        out_sha, sha_feat = self.head_sha(f_share)

        # get the probability of the pred
        prob_sha = torch.softmax(out_sha, dim=1)[:, 1]
        prob_spe = torch.softmax(out_spe, dim=1)[:, 1]

        # build the prediction dict for each output
        pred_dict = {
            'cls': out_sha, 
            'prob': prob_sha, 
            'feat': sha_feat,
            'cls_spe': out_spe,
            'prob_spe': prob_spe,
            'feat_spe': spe_feat,
            'feat_content': content_features,
            'reconstruction_audios': (
                reconstruction_audio_1,
                reconstruction_audio_2,
                self_reconstruction_audio_1, 
                self_reconstruction_audio_2
            )
        }
        return pred_dict

def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )

def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        # self.l1 = nn.Linear(num_classes, in_channel*4, bias=True) #bias is good :)

    def c_norm(self, x, bs, ch, eps=1e-7):
        # assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0)==y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out =   ((x - x_mean.expand(size)) / x_std.expand(size)) \
                * y_std.expand(size) + y_mean.expand(size)
        return out

class Conditional_UNet_Audio(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self):
        super(Conditional_UNet_Audio, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        #self.dropout_half = HalfDropout(p=0.3)
        
        self.adain3 = AdaIN()

        self.dconv_up3a = nn.Conv2d(256, 512, 1, 1)
        self.dconv_up3 = r_double_conv(512, 1024)
        self.conv_last = nn.Conv2d(1024, 1024, 1, 1)
        self.up_last = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.Tanh()
        self.init_weight() 
        
    def forward(self, c, x):  # c is the style and x is the content
        # c = torch.squeeze(c)
        # import pdb;pdb.set_trace()
        # x=torch.unsqueeze(x, dim=2)
        # x=torch.unsqueeze(x, dim=3)
        # [16, 1024]
        b = x.shape[0]
        x = x.reshape(b, 256, 2, 2)
        c = c.reshape(b, 512, 2, 2)
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3a(x) # Do the Dimension Alignment!
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)

        x = self.conv_last(x)
        out = self.up_last(x)

        out = self.activation(out)
        b = out.shape[0]
        out = out.reshape(b, -1)

        return out

class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x

class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x

class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat
