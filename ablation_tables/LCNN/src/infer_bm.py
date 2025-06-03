import os
import os.path as osp
import numpy as np
import pandas as pd
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from feature import calc_cqt, calc_cqt_file, calc_stft, calc_stft_file, load_cqt_from_file
from metrics import calculate_eer
from model.lcnn import build_lcnn

import argparse
import gc

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--iptfl', type=str, help='params helper.')
parser.add_argument('--modelfl', type=str, help='params helper.')

args = parser.parse_args()

feature_type = "cqt"

if __name__ == "__main__":

    if feature_type == "stft":
        print("Extracting data...")
        x, y = calc_stft_file(args.iptfl)
        
    elif feature_type == "cqt":

        print('load ..', end=' ')
        try:
            x = np.load(f"{args.iptfl.replace('.list', '')}.X.npy")
        except:
            x = np.load(f"{args.iptfl}.X.npy")
            
        print('x_train ..', end=' ')
        try:
            y = np.load(f"{args.iptfl.replace('.list', '')}.Y.npy")
        except:
            y = np.load(f"{args.iptfl}.Y.npy")

    input_shape = x.shape[1:]
    lcnn = build_lcnn(input_shape)
    lcnn.load_weights(args.modelfl)

    preds = lcnn.predict(x)

    score = preds[:, 0] - preds[:, 1]  # Get likelihood
    eer = calculate_eer(y, score)  # Get EER score
    print(f"EER : {eer*100} %")
