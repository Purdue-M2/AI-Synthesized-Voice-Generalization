# Improving Generalization for AI-Synthesized Voice Detection

Hainan Ren, Li Lin, Chun-Hao Liu, Xin Wang, Shu Hu
_________________

This repository is the official implementation of our paper "Improving Generalization for AI-Synthesized Voice Detection", which has been accepted by **AAAI 2025**.

<div align="center">
<img src=resources/logo.webp width="50%"/>
</div>

<p align="center">
📚 View the <a href="https://arxiv.org/abs/xxxx" target="_blank">paper</a>
</p>
<p align="center">

## Project Updates

- [x] Checkpoints
- [ ] Training Dataset Protobuf
- [ ] Inference Code



## Quick Start

### Dataset

We trained and tested our model in [LibriSeVoc](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts), [ASVspoof2019](https://www.asvspoof.org/index2019.html), [ASVspoof2021](https://www.asvspoof.org/index2021.html), and [WaveFake](https://github.com/RUB-SysSec/WaveFake). Please Download these dataset.

Here we give a tiny example, refer to `train.list`.

### Checkpoints

[AudioDeepFakeDetection-ckpt-28](https://drive.google.com/file/d/1J8defEI-JJmJVMq4iVlTh825UnyZugQo/view?usp=sharing)

## Method

### Motivation

<div align="center">
<img src=resources/intro-full.png width="100%"/>
</div>

Complex Entangled Information. The generalization issue in AI-synthesized voice detection arises from two main factors. Firstly, many detectors overly emphasize irrelevant content, like identity and background noise. Secondly, different forgery techniques produce unique artifacts, as shown in Figure 2 (Left). The red circle shows unique large-scale artifacts in various AI-synthesized voices, easily detected by a vocoder-specific detector. However, detectors may become overly specialized in specific forgery, hindering generalization to unseen forgeries. To support this hypothesis, we extract features from the LibriSeVoc (Sun et al. 2023)
dataset using RawNet2 (Tak et al. 2021) and Sun et al. (Sun
et al. 2023). The UMAP visualizations (McInnes, Healy,


and Melville 2018) show forgery vocoders’ data clustering closely within baseline feature distribution (Figure 2 Middle (a) and (b)), while different vocoders’ data exhibit more distinct separations. This phenomenon is also observed in the domain-specific feature distribution from our method (Figure 2 (Middle (c)). Moreover, generalizable detectors should treat domain features from AI-synthesized voices equally but distinguish them from features of human voices (as illustrated in Figure 2 (Middle (d))). They should also treat content features equally, whether they are from human voices or AI-synthesized ones (as shown in Figure 2 (Middle (e))).

Sharpness of Loss Landscape. Existing DNN-based AI-synthesized voice detection models, such as RawNet2 (Taket al. 2021) and Sun et al (Sun et al. 2023), are highly over-parameterized and tend to memorize data patterns during training. This results in sharp loss landscapes with multiple minima (Figure2 Right). Such sharpness presents challenges for models to locate the correct global minima for better generalization. Flattening the loss landscape is crucial to smooth the optimization path and enable robust generalization.


### Results

|               Methods              |   LibriSeVoc   |               |                |                |                |                |                |                |  ASVspoof2019  |               |                |                |               |                |                |                |
|:----------------------------------:|:--------------:|:-------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:-------------:|:--------------:|:--------------:|:-------------:|:--------------:|:--------------:|:--------------:|
|                                    |  Seen vocoder  |               |                |                | Unseen vocoder |                |                |                |  Seen vocoder  |               |                | Unseen vocoder |               |                |                |                |
|                                    |       Avg      |      LSV      |       ASP      |       WF       |       Avg      |       ASP      |      FAVC      |       WF       |       Avg      |      ASP      |       LSV      |       Avg      |      ASP      |      FAVC      |       LSV      |       WF       |
|   LCNN{lavrentyeva2019stc}   |      34.04     |      7.80     |      46.21     |      48.10     |      45.08     |      41.90     |      49.28     |      44.06     |      33.02     |     16.15     |      49.88     |      41.21     |      9.81     |      50.98     |      51.93     |      52.13     |
|      RawNet2{tak2021end}     |      20.21     |      1.59     |      29.86     |      29.18     |      30.16     |      24.09     |      33.92     |      32.47     |      20.79     |      1.89     |      39.68     |      39.55     |      6.46     |      51.37     |      47.71     |      52.65     |
|     WavLM{chen2022wavlm}     |      27.26     |     14.12     |      32.88     |      34.79     |      29.54     |      27.18     |      25.64     |      35.80     |      50.75     |     14.22     |      87.28     |      59.27     |      7.91     |      83.84     |      87.28     |      58.07     |
|      XLS-R{babu2021xls}      |      33.47     |     11.21     |      45.37     |      43.83     |      45.11     |      51.23     |      42.91     |      41.18     |      53.65     |      9.04     |      98.26     |      74.85     |      6.40     |      94.91     |      98.26     |      99.82     |
| Sun{sun2023ai} |          18.67 |      3.79     |      22.77     |      29.45     |          27.86 |      24.42     |      25.52     |      33.65     |      22.18     |      3.92     |      40.44     |      41.59     |      8.38     |      54.78     |      50.59     |      52.62     |
|            Ours           | 13.55 | 0.30 | 15.66 | 24.69 | 20.27 | 16.29 | 18.02 | 26.50 | 20.39 | 1.55 | 39.23 | 38.20 | 5.72 | 48.43 | 46.42 | 52.24 |

## Citation
Please kindly consider citing our paper in your publications. 
```bash
@inproceedings{Ren2025improving,
  title={Improving Generalization for AI-Synthesized Voice Detection},
  author={Ren, Hainan and Lin, Li and Liu, Chun-Hao and Wang, Xin and Hu, Shu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
