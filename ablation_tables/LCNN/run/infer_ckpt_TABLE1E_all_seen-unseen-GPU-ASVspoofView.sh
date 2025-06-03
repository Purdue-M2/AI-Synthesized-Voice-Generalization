set -ex

cd ..

# ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/LCNN/n0t0-partially-melgan/lcnn.pth
# ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/LCNN/n0t0/lcnn.pth
# ckpt=/face/hnren/3.SSL/data/research_proposal/benchmark_results/logs_final/LCNN/n0t1-gpu_lr0.001/lcnn.cb.pth
ckpt=$1
gpuid=$2
export CUDA_VISIBLE_DEVICES=${gpuid}

function infer_bmfl(){
    local iptfl=$1
    local ckpt=$2
    local name=$3
    mkdir ${ckpt}.est.Table1E-ASV||true

    python src/infer_bm.py \
        --iptfl ${iptfl} \
        --modelfl ${ckpt} >> ${ckpt}.est.Table1E-ASV/${name}.log 2>&1

}

ipt=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen_LCNN_cqt/asv-view.LibriSeVoc.Seen.test.list
infer_bmfl ${ipt} ${ckpt} seen-librisevoc

ipt=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen_LCNN_cqt/asv-view.ASVspoof2019.Seen.test.list
infer_bmfl ${ipt} ${ckpt} seen-asvspoof


ipt=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen_LCNN_cqt/asv-view.LibriSeVoc.Unseen.test.list
infer_bmfl ${ipt} ${ckpt} unseen-librisevoc

ipt=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen_LCNN_cqt/asv-view.ASVspoof2019.Unseen.test.list
infer_bmfl ${ipt} ${ckpt} unseen-asvspoof


ipt=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/FakeAVCeleb_v1.2/audio/release/LCNN/cqt/test.list
infer_bmfl ${ipt} ${ckpt} unseen-fakeavceleb


ipt=/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/ASVspoof2019/LA/A0/release/seen_unseen_LCNN_cqt/asv-view.WaveFake.Unseen.test.list
infer_bmfl ${ipt} ${ckpt} unseen-wavefake
