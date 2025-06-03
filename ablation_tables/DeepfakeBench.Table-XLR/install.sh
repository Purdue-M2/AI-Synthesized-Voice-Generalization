#!/bin/bash
set -ex
pip install requests -i https://mirror.baidu.com/pypi/simple
pip install cmake -i https://mirror.baidu.com/pypi/simple
pip install numpy==1.21.5  -i https://mirror.baidu.com/pypi/simple
pip install pandas==1.4.2  -i https://mirror.baidu.com/pypi/simple
pip install Pillow==9.0.1  -i https://mirror.baidu.com/pypi/simple
pip install dlib==19.24.0  -i https://mirror.baidu.com/pypi/simple
pip install imageio==2.9.0  -i https://mirror.baidu.com/pypi/simple
pip install imgaug==0.4.0  -i https://mirror.baidu.com/pypi/simple
pip install tqdm==4.61.0 -i https://mirror.baidu.com/pypi/simple
pip install scipy==1.7.3 -i https://mirror.baidu.com/pypi/simple
pip install seaborn==0.11.2  -i https://mirror.baidu.com/pypi/simple
pip install pyyaml==6.0  -i https://mirror.baidu.com/pypi/simple
pip install imutils==0.5.4 -i https://mirror.baidu.com/pypi/simple
pip install opencv-python==4.6.0.66  -i https://mirror.baidu.com/pypi/simple
pip install scikit -image==0.19.2 -i https://mirror.baidu.com/pypi/simple
pip install scikit-learn==1.0.2  -i https://mirror.baidu.com/pypi/simple
pip install albumentations==1.1.0  -i https://mirror.baidu.com/pypi/simple
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra -index-url https://download.pytorch.org/whl/cu113  -i https://mirror.baidu.com/pypi/simple
pip install efficientnet-pytorch==0.7.1  -i https://mirror.baidu.com/pypi/simple
pip install timm==0.6.12 -i https://mirror.baidu.com/pypi/simple
pip install segmentation-models-pytorch==0.3.2 -i https://mirror.baidu.com/pypi/simple
pip install torchtoolbox==0.1.8.2  -i https://mirror.baidu.com/pypi/simple
pip install tensorboard==2.10.1  -i https://mirror.baidu.com/pypi/simple