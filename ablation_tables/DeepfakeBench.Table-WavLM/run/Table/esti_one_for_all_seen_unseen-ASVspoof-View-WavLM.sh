
set -ex
source FUC.sh


function infer_give_ckpt(){
    local ckpt=$1
    local gpuid=$2
    local name=$3
    local testfl=$4

    cd /face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/TABLE/DeepfakeBench.Table-WavLM/training

    CFGfl=config/detector/ucf_audio_${name}_${gpuid}_test.yaml
    cp config/detector/ucf_base.yaml ${CFGfl}
    logfd=${ckpt}.Estimate-ASV.Seen-Unseen.${name}
    touch_fd ${logfd}

    sed "s|log_dir: xxx|log_dir: '${logfd}'|g" -i ${CFGfl}
    sed "s|test_batchSize: 32|test_batchSize: 8|g" -i ${CFGfl}
    sed -i 's/test_dataset: \[FaceShifter, Celeb-DF-v1, DeeperForensics-1.0,]/test_dataset: [FaceShifter, Celeb-DF-v1, DeeperForensics-1.0, LibriSeVoc,]/' ${CFGfl}
    sed "s|test_file: '/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/test-ucf.list'|test_file: '${testfl}'|g" -i ${CFGfl}

    # WavLM dim adaption!
    sed "s|nb_fc_node: 1024|nb_fc_node: 156672|g" -i ${CFGfl}

    export CUDA_VISIBLE_DEVICES=${gpuid}
    python test.py \
        --detector_path ${CFGfl} \
        --test_dataset ${name} \
        --weights_path ${ckpt} 2>&1 >> ${logfd}/table.log

}

function infer_seen_unseen(){
    local ckpt=$1
    local gpuid=$2

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen/asv-view.LibriSeVoc.Seen.test.list
    name=seen-librisevoc
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen/asv-view.ASVspoof2019.Seen.test.list
    name=seen-asvspoof
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen/asv-view.LibriSeVoc.Unseen.test.list
    name=unseen-librisevoc
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}


    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen/asv-view.ASVspoof2019.Unseen.test.list
    name=unseen-asvspoof
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/FakeAVCeleb_v1.2/audio/release/test.list
    name=unseen-fakeavceleb
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

    testfl=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen/asv-view.WaveFake.Unseen.test.list
    name=unseen-wavefake
    infer_give_ckpt ${ckpt} ${gpuid} ${name} ${testfl}

}


ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-WavLM-tb0-ASV/ucf_resnet_audio_lr0.0001_sgd_loss_lambda0.5/ucf_2023-10-31-18-57-54/test/LibriSeVoc/ckpt_26.pth
infer_seen_unseen ${ckpt} 4 &


ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-WavLM-tb0-ASV/ucf_resnet_audio_lr0.0001_sgd_loss_lambda0.5/ucf_2023-10-31-18-57-54/test/LibriSeVoc/ckpt_22.pth
infer_seen_unseen ${ckpt} 5 &

ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-WavLM-tb0-ASV/ucf_resnet_audio_lr0.0001_sgd_loss_lambda0.5/ucf_2023-10-31-18-57-54/test/LibriSeVoc/ckpt_34.pth
infer_seen_unseen ${ckpt} 6 &

# wait
ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/table-WavLM-tb0-ASV/ucf_resnet_audio_lr0.0001_sgd_loss_lambda0.5/ucf_2023-10-31-18-57-54/test/LibriSeVoc/ckpt_25.pth
infer_seen_unseen ${ckpt} 7 &
wait
# wait
