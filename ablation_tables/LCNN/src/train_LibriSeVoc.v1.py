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
parser.add_argument('--svfold', type=str, help='params helper.')
parser.add_argument('--trnfl', type=str, help='params helper.')

args = parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------------------------------
# ✅model parameters
# epochs = 100
epochs = 45
batch_size = 256
lr = 0.00001

# We can use 2 types of spectrogram that extracted by using FFT or CQT.
# Set cqt of stft.
feature_type = "cqt"

# The path for saving model
# This is used for ModelChecking callback.
saving_path = "lcnn.h5"
# ---------------------------------------------------------------------------------------------------------------------------------------


# TEST_FL =  '/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/release-partial-parts/LCNN/cqt/test-ucf.list'
DEV_FL = '/face/hnren/3.SSL/codes/research_proposal/audio_deepfake_detection/codes/scripts/LibriSeVoc/release-partial-parts/LCNN/cqt/dev-ucf.list'
os.makedirs(args.svfold, exist_ok = True)

if __name__ == "__main__":

    class MyCustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            keras.backend.clear_session()
            
    mcc = MyCustomCallback()

    if feature_type == "stft":
        print("Extracting train data...")
        x_train, y_train = calc_stft_file(args.trnfl)
        print("Extracting dev data...")
        x_val, y_val = calc_stft_file(DEV_FL)
        
    elif feature_type == "cqt":
        print("Extracting train data...")
        x_train, y_train = load_cqt_from_file(args.trnfl)
        # print("Extracting dev data...")
        # x_val, y_val = load_cqt_from_file(DEV_FL)

    input_shape = x_train.shape[1:]
    lcnn = build_lcnn(input_shape)

    lcnn.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Session keyword arguments are not support during eager execution. You passed: {'run_eagerly': True}
    
    # Callbacks
    # es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    # cp_cb = ModelCheckpoint(
    #     filepath=args.svfold,
    #     monitor="val_loss",
    #     verbose=1,
    #     save_best_only=True,
    #     mode="auto",
    #     )

    # Train LCNN
    history = lcnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=None,
        callbacks=[mcc],
    )
    del x_train, y_train
    

    lcnn.save(f'{args.svfold}/lcnn.pth')  # The file needs to end with the .keras extension

    # print("Extracting eval data")
    # df_eval = pd.read_csv(protocol_eval)

    # if feature_type == "stft":
    #     x_eval, y_eval = calc_stft(df_eval, path_eval)

    # elif feature_type == "cqt":
    #     x_eval, y_eval = calc_cqt(df_eval, path_eval)

    # # predict
    # preds = lcnn.predict(x_eval)

    # score = preds[:, 0] - preds[:, 1]  # Get likelihood
    # eer = calculate_eer(y_eval, score)  # Get EER score
    # print(f"EER : {eer*100} %")
