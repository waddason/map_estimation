import rampwf as rw

import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

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
    cv = GroupShuffleSplit(n_splits=1, test_size=1, random_state=37298)
    return cv.split(X, y, groups=X["subject"])


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

    file = 'train.h5'
    file = Path(path) / "data" / file
    if os.environ.get("RAMP_TEST_MODE", False):
        X_v, y_v = _load_data(file, 0, 1000, load_waveform=load_waveform)
        X_m, y_m = _load_data(file, -1001, -1, load_waveform=load_waveform)
        rw.X_TRAIN = pd.concat([X_v, X_m], axis=0)
        rw.Y_TRAIN = np.concatenate([y_v, y_m], axis=0)
    else:
        rw.X_TRAIN, rw.Y_TRAIN = _load_data(file, start, stop, load_waveform)
    return rw.X_TRAIN, rw.Y_TRAIN


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
