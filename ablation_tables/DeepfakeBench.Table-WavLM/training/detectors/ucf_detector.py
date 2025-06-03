'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the UCFDetector

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
import copy

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='ucf')
class UCFDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['backbone_config']['num_classes']
        self.encoder_feat_dim = config['encoder_feat_dim']
        self.half_fingerprint_dim = self.encoder_feat_dim//2
        

        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.header_realfake = nn.Linear(in_features = config['backbone_config']['nb_fc_node'], # 1024
			out_features = config['backbone_config']['nb_classes'],bias=True) # 2
        self.header_vocoder = nn.Linear(in_features = config['backbone_config']['nb_fc_node'], # 1024
			out_features = config['backbone_config']['nb_vocoders'],bias=True) # 7
        
        sv_cfg = config['backbone_config']
        # _ckpt = torch.load(config['weights_path'], map_location='cpu')
        # _cfg = copy.deepcopy(_ckpt['cfg'])
        # del _ckpt
        _cfg = {'extractor_mode': 'default',
        'encoder_layers': 12,
        'encoder_embed_dim': 768,
        'encoder_ffn_embed_dim': 3072,
        'encoder_attention_heads': 12,
        'activation_fn': 'gelu',
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.0,
        'encoder_layerdrop': 0.05,
        'dropout_input': 0.1,
        'dropout_features': 0.1,
        'layer_norm_first': False,
        'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2',
        'conv_bias': False,
        'feature_grad_mult': 0.1,
        'mask_length': 10,
        'mask_prob': 0.8,
        'mask_selection': 'static',
        'mask_other': 0.0,
        'no_mask_overlap': False,
        'mask_min_space': 1,
        'mask_channel_length': 10,
        'mask_channel_prob': 0.0,
        'mask_channel_selection': 'static',
        'mask_channel_other': 0.0,
        'no_mask_channel_overlap': False,
        'mask_channel_min_space': 1,
        'conv_pos': 128,
        'conv_pos_groups': 16,
        'relative_position_embedding': True,
        'num_buckets': 320,
        'max_distance': 800,
        'gru_rel_pos': True,
        'expand_attention_head_size': -1}
        _cfg['dropout'] = 0.0
        _cfg['dropout_features'] = 0.0
        _cfg['dropout_input'] = 0.0
        _cfg['dropout_features'] = 0.0
        
        from networks.WavLM import WavLMConfig
        config['backbone_config'] = WavLMConfig(_cfg)

        self.encoder = self.build_backbone(config) # the `forgery` feature should be furthur decoupled!
        
        self.loss_lambda = float(config['loss_func']['loss_lambda'])

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

        cls_loss_func = cls_loss_class()
        spe_loss_func = spe_loss_class()

        loss_func = {
            'cls': cls_loss_func, 
            'spe': spe_loss_func,
        }
        return loss_func
    
    def features(self, data_dict: dict, inference=False) -> torch.tensor:
        cat_data = data_dict['audio']
        # encoder
        feat = self.encoder.features(cat_data, inference = inference)
        feat_dict = {'feat': feat}
        return feat_dict

    def classifier(self, features: torch.tensor) -> torch.tensor:
        ...

    def get_losses(self, data_dict: dict, pred_dict: dict, inference=False) -> dict:
        # if 'label_spe' in data_dict and 'reconstruction_audios' in pred_dict and not inference:
        if not inference:
            return self.get_train_losses(data_dict, pred_dict)
        else:  # test mode
            return self.get_test_losses(data_dict, pred_dict)

    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get combined, real, fake imgs
        cat_data = data_dict['audio']
        real_audio, fake_audio = cat_data.chunk(2, dim=0)

        # get label
        label = data_dict['label']
        label_spe = data_dict['label_spe']
        # get pred
        pred = pred_dict['cls']
        pred_spe = pred_dict['cls_spe']
        # 1. classification loss for common features
        loss_rf = self.loss_func['cls'](pred, label) # predict the Real or Fake
        loss_spe = self.loss_func['spe'](pred_spe, label_spe)
        
        loss = loss_rf * self.loss_lambda + loss_spe * (1-self.loss_lambda) 
        
        loss_dict = {
            'overall': loss,
            'classfication': loss,
        }
        return loss_dict

    def get_test_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get label
        label = data_dict['label']
        # get pred
        pred = pred_dict['cls']
        # for test mode, only classification loss for common features
        loss = self.loss_func['cls'](pred, label)
        loss_dict = {'classfication': loss}
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

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        # import pdb;pdb.set_trace()
        if self.prob and self.label:
            if len(self.prob[-1].shape) == 0:
                self.prob[-1] = np.expand_dims(self.prob[-1], axis=0)
                self.label[-1] = np.expand_dims(self.label[-1], axis=0)
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
        feat_dict = self.features(data_dict, inference = inference)
        feat = feat_dict['feat']
        b = feat.shape[0]
        feat = feat.reshape(b, -1)
        out_rf = self.header_realfake(feat)
        out_spe = self.header_vocoder(feat)

        if inference:
            out = out_rf
            # inference only consider share loss
            prob = torch.softmax(out, dim=1)[:, 1]
            self.prob.append(
                prob
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
            _, prediction_class = torch.max(out, 1)
            correct = (prediction_class == data_dict['label']).sum().detach().cpu().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

            pred_dict = {'cls': out.detach(), 'feat': feat.detach().squeeze().cpu().numpy()}
            return  pred_dict


        # get the probability of the pred
        prob_rf = torch.softmax(out_rf, dim=1)[:, 1]
        prob_spe = torch.softmax(out_spe, dim=1)[:, 1]

        # build the prediction dict for each output
        pred_dict = {
            'cls': out_rf, 
            'cls_spe': out_spe, 
            'prob_rf': prob_rf, 
            'prob_spe': prob_spe, 
            'feat': None,
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

