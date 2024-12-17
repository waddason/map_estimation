import rampwf as rw

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

problem_title = 'MAP estimation from non-invasive monitoring'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()

# An object implementing the workflow
workflow = rw.workflows.Estimator()


class MAE(rw.score_types.BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mae', precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mask = y_true != -1
        return np.mean(np.abs((y_true - y_pred)[mask]))


score_types = [
    MAE(name='mean_absolute_error', precision=5),
]


def get_cv(X, y):
    # Make sure the index is a range index so it is compatible with sklearn API
    X = X.reset_index(drop=True)

    chunks = X['chunk'].fillna('t')

    def split():
        train_idx = chunks[chunks != 'val'].index
        val_idx = chunks[chunks == 'val'].index
        yield train_idx, val_idx

    return split()


def _load_data(file, start=None, stop=None, load_waveform=True):
    X_df = pd.read_hdf(file, key="data", start=start, stop=stop)

    y = X_df['map']
    X_df = X_df.drop(columns=['map', 'sbp', 'dbp'], errors='ignore')

    if load_waveform:
        with h5py.File(file, 'r') as f:
            X_df['ecg'] = list(f['ecg'][start:stop])
            X_df['ppg'] = list(f['ppg'][start:stop])

    # Replace None value in y by `-1
    y = y.fillna(-1).values

    return X_df, y


# READ DATA
def get_train_data(path='.', start=None, stop=None, load_waveform=True):

    # Avoid loading the data if it is already loaded
    # We use a global variable in rw as the problem.py module is created
    # dynamically and the global variables are not always reused.
    hash_train = hash((str(path), start, stop, load_waveform))
    if getattr(rw, "HASH_TRAIN", -1) == hash_train:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TRAIN = hash_train

    train_file = Path(path) / "data" / 'train.h5'
    val_file = Path(path) / "data" / 'validation.h5'
    if os.environ.get("RAMP_TEST_MODE", False):
        start_s, stop_s = 0, 1000
        start_t, stop_t = -1001, -1
        start_val, stop_val = 0, 100
    else:
        start_s, stop_s = 0, int(1.5e5)
        start_t, stop_t = -int(1.5e5+1), -1
        start_val, stop_val = None, None
    X_s, y_s = _load_data(train_file, start_s, stop_s, load_waveform)
    X_t, y_t = _load_data(train_file, start_t, stop_t, load_waveform)
    X_val, y_val = _load_data(val_file, start_val, stop_val, load_waveform)
    X_val['chunk'] = 'val'
    X_train = pd.concat([X_s, X_t, X_val], axis=0, ignore_index=True)
    y_train = np.concatenate([y_s, y_t, y_val], axis=0)

    rw.X_TRAIN, rw.Y_TRAIN = X_train, y_train
    return X_train, y_train


def get_test_data(path='.', start=None, stop=None, load_waveform=True):

    hash_test = hash((str(path), start, stop, load_waveform))
    if getattr(rw, "HASH_TEST", -1) == hash_test:
        return rw.X_TRAIN, rw.Y_TRAIN

    rw.HASH_TEST = hash_test

    file = 'test.h5'
    file = Path(path) / "data" / file
    if os.environ.get("RAMP_TEST_MODE", False):
        start, stop = 0, 100
    rw.X_TEST, rw.Y_TEST = _load_data(file, start, stop, load_waveform)
    return rw.X_TEST, rw.Y_TEST
