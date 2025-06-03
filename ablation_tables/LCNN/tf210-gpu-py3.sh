#!/bin/bash
image=tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

docker pull $image

docker run --runtime=nvidia --rm -ti --privileged $@ \
    -v $PWD:/workspace -w /workspace -v /ssd:/ssd \
    -v /tracking:/tracking \
    -v /training:/training -v /face:/face \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /var/run/docker.sock:/var/run/docker.sock  \
    -e PATH=/workspace/cluster/bin/:$PATH \
    --shm-size=7gb \
    $image \
    bash
