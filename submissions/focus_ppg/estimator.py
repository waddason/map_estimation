# This estimator focuses on the data from the PPG
# In the spirit of non-invasive method to get the map
# drop the ECG and possible empty columns
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

from skada import CORALAdapter, make_da_pipeline, OTLabelPropAdapter

keep_col = ["age", "gender", "ppg"]


def extract_ppg_features(signal, sampling_rate=125, nb_out_freq=3):
    """Extract the relevant features of ppg signal using Fourier transform."""
    T = 1.0 / sampling_rate  # Sampling interval
    # Perform Fourier Transform
    yf = fft(signal)
    xf = fftfreq(sampling_rate, T)[: sampling_rate // 2]
    # Extract features
    dominant_frequencies = xf[
        np.argsort(-np.abs(yf[: sampling_rate // 2]))[:nb_out_freq]
    ]
    # print(dominant_frequencies)
    dominant_amplitudes = (
        2.0
        / sampling_rate
        * np.abs(
            yf[np.argsort(-np.abs(yf[: sampling_rate // 2]))[:nb_out_freq]]
        )
    )
    return np.concatenate([dominant_frequencies, dominant_amplitudes])


def prepare_data(X, nb_out_freq, chunk_size=1000):
    """Prepare the input to feed the estimator."""
    print("Transform the input data:")

    # Initialize transformers
    imp_med = SimpleImputer(strategy="median")
    gend_cols = ["M", "F"]
    ohe = OneHotEncoder(
        categories=[gend_cols],
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    # Age
    X_age = imp_med.fit_transform(X[["age"]])
    print(f"- Has filled {np.isnan(X['age']).sum()} age missing values.")

    # Gender
    X_gender = ohe.fit_transform(X[["gender"]])
    print(
        f"- Has classified {X.shape[0]} lines into {X_gender.shape[1]} gender categories."
    )
    # Process in chunks to reduce memory usage
    chunks = []
    fft_columns = [f"fq_{i:02d}" for i in range(nb_out_freq)] + [
        f"am_{i:02d}" for i in range(nb_out_freq)
    ]
    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        chunk = X.iloc[start:end]
        # PPG
        ppg_features = chunk["ppg"].apply(
            lambda signal: extract_ppg_features(signal, nb_out_freq=nb_out_freq)
        )
        X_ppg_features = pd.DataFrame(
            ppg_features.tolist(), index=chunk.index, columns=fft_columns
        )

        # Concatenate the processed chunk
        chunks.append(X_ppg_features)

    # Merge with other features
    all_ppg_features = pd.concat(chunks, axis=0)
    print(
        f"- Extracted {all_ppg_features.shape[1]} PPG features for {all_ppg_features.shape[0]} samples."
    )
    X_processed = pd.concat(
        [
            pd.DataFrame(X_age, index=X.index, columns=["age"]),
            pd.DataFrame(X_gender, index=X.index, columns=gend_cols),
            all_ppg_features,
        ],
        axis="columns",
    )
    return X_processed


class MyEstimator:
    def __init__(self, nb_out_freq=3):
        self.nb_out_freq = (
            nb_out_freq  # number of main frequencies to keep after FFT
        )
        self.pipe = make_da_pipeline(
            # Keep only 5 columns
            CORALAdapter(),
            # OTLabelPropAdapter(),
            HistGradientBoostingRegressor(
                loss="absolute_error",
            ),
        )

    def fit(self, X, y):
        # # keep only the labeled data:
        X_train = X[y != -1][keep_col + ["domain"]]
        y_train = y[y != -1]
        # X_train = X[keep_col + ["domain"]]
        # y_train = y
        # Transform the domain for use with skada estimator
        sample_domain = X_train["domain"].map({"v": 1, "m": -1})
        X_train = X_train.drop(columns=["domain"])
        print(f"Domain {sample_domain.value_counts()}")
        X_prepared = prepare_data(X_train, self.nb_out_freq)

        print(f"\nFit on {X_prepared.columns.to_list()} {X_prepared.shape}")
        print(X_prepared.head())
        self.pipe.fit(X_prepared, y_train, sample_domain=sample_domain)
        print("fit done with", self.pipe)
        return self

    def predict(self, X):
        # Issue: missing columns on test data for prediction.
        # Solution: drop bmi/age/weight
        X_prepared = prepare_data(X[keep_col], self.nb_out_freq)
        print(f"\nPredict on {X_prepared.columns.to_list()} {X_prepared.shape}")

        return self.pipe.predict(X_prepared)


def get_estimator():
    model = MyEstimator(nb_out_freq=6)
    return model
