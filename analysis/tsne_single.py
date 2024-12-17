import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle

FAKE_AUDIO_VOCODER=['melgan', 'parallel_wave_gan', 'waveglow', 'mb_mel_gan', 'fb_mel_gan', 'hifi_gan']
color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']
label_dict = {k+1:v for k,v in enumerate(FAKE_AUDIO_VOCODER)}
label_dict[0] = 'gt'

def tsne_draw(x_transformed, numerical_labels, ax, epoch=0, log='', detector_name=None, legend=None):
    labels = [label_dict[label] for label in numerical_labels]

    tsne_df = pd.DataFrame(x_transformed, columns=['X', 'Y'])
    tsne_df["Targets"] = labels
    tsne_df["NumericTargets"] = numerical_labels
    tsne_df.sort_values(by="NumericTargets", inplace=True)
    
    marker_list = ['*' if label == 0 else 'o' for label in tsne_df["NumericTargets"]]

    for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], [color_map[i] for i in tsne_df["NumericTargets"]], marker_list):
        if legend:
            ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m, label=legend)
        else:
            ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m)

    print(f'epoch{epoch} ' + log)
    ax.axis('off')

CKPT_EST_FD='/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/n0t2/ucf_resnet_audio_lr0.0002_adam/ucf_2023-10-05-22-59-34/test/LibriSeVoc/ckpt_27.pth.Estimate.WaveFake'

detector_name_list = [f'{CKPT_EST_FD}/tsne-{e}.pkl' for e in FAKE_AUDIO_VOCODER]

tsne = TSNE(n_components=2, perplexity=20, random_state=1024, learning_rate=250)
# fig, axs = plt.subplots(1, len(FAKE_AUDIO_VOCODER), figsize=(20,10))
# import pdb;pdb.set_trace()
fig, axs = plt.subplots(1, 1, figsize=(20,10))

for i, tsne_dict in enumerate(detector_name_list):
    print(f'Processing {tsne_dict}...')
    name = str(tsne_dict.split('/')[-1].split('.')[0].split('_')[-1])
    with open(tsne_dict, 'rb') as f:
        tsne_dict = pickle.load(f)
    
    feat = tsne_dict['feat'].reshape((tsne_dict['feat'].shape[0], -1))
    label_spe = tsne_dict['label_spe']

    label_0_indices = np.where(label_spe == 0)[0][:2500]
    other_label_indices = np.where(label_spe != 0)[0]
    num_samples = len(label_0_indices)
    other_label_indices_sampled = np.random.choice(other_label_indices, size=num_samples, replace=False)
    sampled_indices = np.concatenate((label_0_indices, other_label_indices_sampled))
    np.random.shuffle(sampled_indices)

    feat = feat[sampled_indices]
    label_spe = label_spe[sampled_indices]
    feat_transformed = tsne.fit_transform(feat)
    g_label_spe = [e if e == 0 else e+i for e in label_spe]
    # try:
    scatter = tsne_draw(feat_transformed, g_label_spe, ax=axs, epoch=0, log='share_in_specific', detector_name='xception', legend=FAKE_AUDIO_VOCODER[i])
    #     vi+=1
    # except:
    #     pass
    # # give a title to the subplot
    # axs[i].set_title(f'Xception with {name} frames')  

# create a legend for the whole figure after the loop
# import pdb;pdb.set_trace()
# handles = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color_map[i], markersize=10) if i == 0 else plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=10) for i in range(7)]
# labels = [label_dict[i] for i in range(7)]
# fig.legend(handles, labels, title="Classes", loc="upper right", fontsize=14)

plt.tight_layout()
savefl = f'{CKPT_EST_FD}/tsne_in_one.png'
plt.savefig(savefl)
print(savefl)