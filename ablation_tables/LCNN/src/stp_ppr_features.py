import os
import os.path as osp
import numpy as np
from feature import build_cqt_map_from_file, build_stft_map_from_file

import argparse

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--ipt_fl', type=str, help='params helper.')
parser.add_argument('--feat_type', type=str, help='params helper.')

args = parser.parse_args()

if __name__ == "__main__":
    if args.feat_type == "stft":
        audp_feat_m = build_stft_map_from_file(args.ipt_fl)
    elif args.feat_type == "cqt":
        audp_feat_m = build_cqt_map_from_file(args.ipt_fl)
    
    for audp, feat in audp_feat_m.items():
        npy_audp = f"{audp}.npy"
        np.save(npy_audp, feat)
    print(f'=> {args.ipt_fl} .. done ..')
