# --- header ---
import os
import os.path as osp
import cv2
import tqdm
import shutil
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle


import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import gridspec

def showcv2(imgpath):
    plt.imshow(cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB), interpolation='nearest')
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def show_identity(imgs,size=(128,384),figsize=(10,3)):
    plt.figure(figsize=figsize)
    imglist=[]
    for img in imgs:
        imglist.append(cv2.resize(cv2.imread(img),size, interpolation=cv2.INTER_CUBIC)[:,:,::-1])
    plt.imshow(np.concatenate(np.array(imglist), axis=1))


def show_identity_v2(imgs,size=(128,384),figsize=(10,3)):
    plt.figure(figsize=figsize)
    Nc = 8
    Nr = len(imgs) // Nc if len(imgs) % Nc == 0 else (len(imgs) // Nc + 1)
    IM = []
    for i in range(Nr):
        _IM = []
        for j in range(Nc):
            if i*Nc + j <len(imgs):
                im = cv2.resize(cv2.imread(imgs[i*Nc+j]), size, interpolation=cv2.INTER_CUBIC)/255.
            else:
                im = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            _IM.append(im)
        
        _IM = np.concatenate(_IM, axis=1)
        IM.append(_IM)

    IM = np.concatenate(IM, axis=0)
    plt.imshow(IM[:,:,::-1])
    
    
FAKE_AUDIO_VOCODER=['melgan', 'parallel_wave_gan', 'waveglow', 'mb_mel_gan', 'fb_mel_gan', 'hifi_gan']
CKPT_EST_FD='/home/work//research_proposal/benchmark_results/logs_final/n0t2/audiofakedetection_reset_audio_lr0.0002_adam/audiofakedetection_reset2023-10-05-22-59-34/test/LibriSeVoc/ckpt_27.pth.Estimate.WaveFake.v2'

detector_name_list = [f'{CKPT_EST_FD}/tsne-{e}.pkl' for e in FAKE_AUDIO_VOCODER]

color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']
label_dict = {k+1:v for k,v in enumerate(FAKE_AUDIO_VOCODER)}
label_dict[0] = 'gt'


class UtTsner(object):
    @classmethod
    def load_map(cls, res_pkl):
        with open(res_pkl, 'rb') as f:
            tsne_dict = pickle.load(f)
        return tsne_dict
    
    @classmethod
    def load_tsne(cls, pper, lr):
        tsne = TSNE(n_components=2, perplexity=pper, random_state=1024, learning_rate=lr)
        return tsne
    
    @classmethod
    def load_samples(cls, tsne_dict, num=-1):
        
        feat_common = tsne_dict['feat'].reshape((tsne_dict['feat'].shape[0], -1))
        feat_specific = tsne_dict['feat_specific'].reshape((tsne_dict['feat_specific'].shape[0], -1))
        feat_content = tsne_dict['feat_content'].reshape((tsne_dict['feat_content'].shape[0], -1))
        label_spe = tsne_dict['label_spe']
        if num != -1:
            label_0_indices = np.where(label_spe == 0)[0][:num]
        else:
            label_0_indices = np.where(label_spe == 0)[0]

        other_label_indices = np.where(label_spe != 0)[0]
        num_samples = len(label_0_indices)
#         np.random.shuffle(sampled_indices)
        other_label_indices_sampled = np.random.choice(other_label_indices, size=num_samples, replace=False)
        sampled_indices = np.concatenate((label_0_indices, other_label_indices_sampled))
        np.random.shuffle(sampled_indices)

        feat_common = feat_common[sampled_indices]
        feat_specific = feat_specific[sampled_indices]
        feat_content = feat_content[sampled_indices]
        label_spe = label_spe[sampled_indices]
        
        return feat_common, feat_specific, feat_content, label_spe
        
    @classmethod
    def build_transform(cls, tsne, feat):
        feat_transformed = tsne.fit_transform(feat)
        return feat_transformed
    
    @classmethod
    def build_df_and_figure(cls, label_spe, si, label_dict, feat_transformed, feat_type):
        g_label_spe = [e if e == 0 else e+si for e in label_spe]

        labels = [label_dict[label] for label in g_label_spe]
        tsne_df = pd.DataFrame(feat_transformed, columns=['X', 'Y'])
        tsne_df["Targets"] = labels
        tsne_df["NumericTargets"] = g_label_spe
        tsne_df.sort_values(by="NumericTargets", inplace=True)
        
        plt.figure(figsize=(10,10))
        
#         marker_list = ['*' if label == 0 else 'o' for label in tsne_df["NumericTargets"]]
#         import pdb;pdb.set_trace()
        for _x, _y, _gtcls in tqdm.tqdm(zip(tsne_df['X'], tsne_df['Y'], tsne_df["NumericTargets"]), total=len(tsne_df['X'])):
            plt.scatter(_x, _y, color=color_map[_gtcls], label=labels[_gtcls])
        plt.title(f"{feat_type} - {osp.basename(detector_name_list[i])}")
        plt.show()

        
    @classmethod
    def build_df_and_figure_v2(cls, label_spe, si, label_dict, feat_transformed, feat_type, workname, a=0.5):
        
        g_label_spe = [e if e == 0 else e+i for e in label_spe]
        
        labels = [label_dict[label] for label in g_label_spe]
        tsne_df = pd.DataFrame(feat_transformed, columns=['X', 'Y'])
        tsne_df["Targets"] = labels
        tsne_df["NumericTargets"] = g_label_spe
        tsne_df.sort_values(by="NumericTargets", inplace=True)


        plt.figure()
        v = 0
        tdf = tsne_df.query(f"NumericTargets == {v}")
        plt.scatter(tdf['X'].tolist(), tdf['Y'].tolist(),s = 1, color=color_map[v], label = label_dict[v])

        v = 1+si
        tdf = tsne_df.query(f"NumericTargets == {v}")
        plt.scatter(tdf['X'].tolist(), tdf['Y'].tolist(),s = 1, alpha=a, color=color_map[v], label = label_dict[v])

        plt.title(f"{feat_type} - {workname}")
        plt.legend() 
        plt.show()
        
        
        
WFOLD=osp.join(CKPT_EST_FD, 'visss')
os.makedirs(WFOLD, exist_ok = True)

for i in tqdm.tqdm(range(6), total=6):
    vocoder = osp.basename(detector_name_list[i])
    tsne_dict = UtTsner.load_map(detector_name_list[i])
    feat_common, feat_specific, feat_content, label_spe = UtTsner.load_samples(tsne_dict)

    tsner = UtTsner.load_tsne(pper=5, lr='auto')
    feat_content_t = UtTsner.build_transform(tsner, feat_content); del tsner
    np.save(f"{WFOLD}/feat_content_t.{vocoder}.npy", feat_content_t)
    
    # UtTsner.build_df_and_figure_v2(label_spe, i, label_dict, feat_content_t, 'CONTENT', workname = osp.basename(detector_name_list[i]), a = 0.5)

    tsner = UtTsner.load_tsne(pper=5, lr='auto')
    feat_common_t = UtTsner.build_transform(tsner, feat_common); del tsner
    # UtTsner.build_df_and_figure_v2(label_spe, i, label_dict, feat_common_t, 'COMMON', workname = osp.basename(detector_name_list[i]), a = 0.2)
    np.save(f"{WFOLD}/feat_common_t.{vocoder}.npy", feat_common_t)
    
    
    tsner = UtTsner.load_tsne(pper=5, lr='auto')
    feat_specific_t = UtTsner.build_transform(tsner, feat_specific); del tsner
    # UtTsner.build_df_and_figure_v2(label_spe, i, label_dict, feat_specific_t, 'SPECIFIC', workname = osp.basename(detector_name_list[i]), a = 0.2)
    np.save(f"{WFOLD}/feat_specific_t.{vocoder}.npy", feat_specific_t)
    
    print(f"==> 🥏 {vocoder} ..")