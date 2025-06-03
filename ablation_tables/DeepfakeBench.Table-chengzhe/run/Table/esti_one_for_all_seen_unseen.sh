
set -ex
source FUC.sh


function infer_give_ckpt(){
    local ckpt=$1
    local gpuid=$2
    local name=$3
    local testfl=$4

    cd /face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/TABLE/DeepfakeBench.Table-chengzhe/training

    CFGfl=config/detector/ucf_audio_${name}_test.yaml
    cp config/detector/ucf_base.yaml ${CFGfl}
    logfd=${ckpt}.Estimate.Seen-Unseen.${name}
    touch_fd ${logfd}

    sed "s|log_dir: xxx|log_dir: '${logfd}'|g" -i ${CFGfl}
    sed "s|test_batchSize: 32|test_batchSize: 32|g" -i ${CFGfl}
    sed -i 's/test_dataset: \[FaceShifter, Celeb-DF-v1, DeeperForensics-1.0,]/test_dataset: [FaceShifter, Celeb-DF-v1, DeeperForensics-1.0, LibriSeVoc,]/' ${CFGfl}
    sed "s|test_file: '/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/test-ucf.list'|test_file: '${testfl}'|g" -i ${CFGfl}

    export CUDA_VISIBLE_DEVICES=${gpuid}
    python test.py \
        --detector_path ${CFGfl} \
        --test_dataset ${name} \
        --weights_path ${ckpt} 2>&1 >> ${logfd}/table.log

}

function infer_seen_unseen(){
    local ckpt=$1
    local gpuid=$2

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/test-ucf.list
    name=librisevoc
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/A0/release/seen_unseen/lsv-view.WaveFake.Seen.test.list
    name=seen-wavefake
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/A0/release/seen_unseen/lsv-view.ASVspoof2019.Seen.test.list
    name=seen-asvpspoof2019
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}


    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/A0/release/seen_unseen/lsv-view.WaveFake.Unseen.test.list
    name=unseen-wavefake
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/A0/release/seen_unseen/lsv-view.ASVspoof2019.Unseen.test.list
    name=unseen-asvspoof2019
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/FakeAVCeleb_v1.2/audio/release/test.list
    name=unseen-fakeavceleb
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}
}


ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0/ucf_resnet_audio_lr_sgd/ucf_2023-10-20-15-54-22/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0-partially/ucf_resnet_audio_lr0.0001_sgd_wo-melgan/ucf_2023-10-20-17-17-19/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0-partially/ucf_resnet_audio_lr0.0001_sgd_wo-wavegrad/ucf_2023-10-20-17-17-54/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0-partially/ucf_resnet_audio_lr0.0001_sgd_wo-diffwave/ucf_2023-10-20-17-17-04/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0-partially/ucf_resnet_audio_lr0.0001_sgd_wo-wavernn/ucf_2023-10-21-01-01-06/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0-partially/ucf_resnet_audio_lr0.0001_sgd_wo-wavenet/ucf_2023-10-21-01-00-51/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-chengzhe-tb0-partially/ucf_resnet_audio_lr0.0001_sgd_wo-parallel_wave_gan/ucf_2023-10-20-17-17-36/test/LibriSeVoc/ckpt_20.pth
infer_seen_unseen ${ckpt} 1
