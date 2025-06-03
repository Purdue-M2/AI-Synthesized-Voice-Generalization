import os
import os.path as osp
import numpy as np

from feature import calc_cqt, calc_cqt_file, calc_stft, calc_stft_file, load_cqt_from_file

import argparse

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--ipt_fl', type=str, help='params helper.')

args = parser.parse_args()

    
if __name__ == "__main__":
        x, y = load_cqt_from_file(args.ipt_fl)
        _save_x_fl = f"{args.ipt_fl}.X.npy"
        _save_y_fl = f"{args.ipt_fl}.Y.npy"
        print(f" 🥏 ==> {_save_x_fl} ..")
        np.save(_save_x_fl, x)
        print(f" 🥏 ==> {_save_y_fl} ..")
        np.save(_save_y_fl, y)
