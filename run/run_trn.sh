set -ex
source FUC.sh
JOBNAME=tst

cd ../training

GPUID=7
export CUDA_VISIBLE_DEVICES=${GPUID}

# lr=0.0002
# opt=adam
# mode=fd
# measure=DV
# mi_lambda=0.03
# rho=0.07
# use_adaptive_sam=False

logfd=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/${JOBNAME}/ucf_resnet_audio_lr${lr}_${opt}_${mode}-${measure}-SAM.rho${rho}.adap-${use_adaptive_sam}
touch_fd ${logfd}

CFGfl=config/detector/ucf_audio.yaml
ckptfl=../ckpt_28.pth


python train.py \
    --detector_path ${CFGfl} \
    --no-save_feat \
    --weights_path ${ckptfl}