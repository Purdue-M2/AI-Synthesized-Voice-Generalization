set -ex
cp sources.list /etc/apt/sources.list
apt update
apt upgrade
apt install libsndfile1
pip install typeguard -i https://mirror.baidu.com/pypi/simple
pip install -r requirements-slim.txt -i https://mirror.baidu.com/pypi/simple