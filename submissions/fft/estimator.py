"""An estimator focuses on the data from the PPG.

In the spirit of non-invasive method to get the map
drop the ECG and possible empty columns
"""  # noqa: INP001

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from skada import CORALAdapter, OTLabelPropAdapter, make_da_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

keep_col = ["age", "gender", "ppg", "ecg"]


class MyRFR(BaseEstimator, TransformerMixin):
    """Custom random forest."""

    def __init__(self, n_estimators: int = 100, random_state: int = 1) -> None:
        """Create a custom random forest regressor for MAP estimation."""
        self.model = make_pipeline(
            make_column_transformer(
                (SimpleImputer(strategy="median"), ["age"]),
                (
                    OneHotEncoder(
                        categories=[["M", "F"]],
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                    ["gender"],
                ),
            ),
            RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state
            ),
        )
        # Filtering param
        self.cutoff = 5.0  # Cutoff frequency in Hz
        self.fs = 125.0  # Sampling frequency in Hz in input data
        self.order = 5

    def compute_heart_rate(self, ecg_signal: np.ndarray[float]) -> float:
        """Compute the heart rate from ECG signal.

        Args:
            ecg_signal (np.ndarray[float]): The ECG signal.

        Returns:
            float: The computed heart rate.

        """
        peaks, _ = find_peaks(ecg_signal, distance=self.fs / 2)
        rr_intervals = np.diff(peaks) / self.fs
        # Heart rate
        return (
            60 / np.nanmean(rr_intervals) if len(rr_intervals) > 0 else np.nan
        )

    def compute_ptt(self, ppg_signal: np.ndarray[float]) -> float:
        """Compute the pulse transit time from PPG signal.

        Args:
            ppg_signal (np.ndarray[float]): The PPG signal.

        Returns:
            float: The computed pulse transit time.

        """
        ppg_peaks, _ = find_peaks(ppg_signal, distance=self.fs / 2)
        # PPT
        return (
            np.nanmean(np.diff(ppg_peaks) / self.fs)
            if len(ppg_peaks) > 1
            else np.nan
        )

    def prepare_x(self, x: pd.DataFrame) -> pd.DataFrame:
        """Use filtering techniques to remove noise from the PPG and ECG.

        Args:
            x (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The prepared data with filtered signals and computed
            features.

        """
        prepared_x = x[["age", "gender"]].copy()
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a, sos = butter(
            self.order,
            normal_cutoff,
            btype="low",
            analog=False,
            output="sos",
        )
        # assert "ppg" in x.columns, "Error, no ppg column, in input data"
        prepared_x["PPG_filtered"] = sosfiltfilt(sos, x["ppg"])
        # assert "ecg" in x.columns, "Error, no ecg column, in input data"
        prepared_x["ECG_filtered"] = sosfiltfilt(sos, x["ecg"])

        # Heart rate from ECG
        prepared_x["heart_rate"] = prepared_x["ECG_filtered"].apply(
            self.compute_heart_rate
        )

        # Pulse transit time (PTT)
        prepared_x["ptt"] = prepared_x["PPG_filtered"].apply(self.compute_ptt)

        return prepared_x.drop(columns=["ECG_filtered", "PPG_filtered"])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MyRFR":
        """Fit the model to the data.

        Args:
            X (pd.DataFrame): The input data.
            y (np.ndarray): The target values.

        Returns:
            MyRFR: The fitted model.

        """
        # Ignore sample from unknown domain
        X = X[y != -1]
        y = y[y != -1]
        self.model.fit(self.prepare_x(X), y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The transformed data.

        """
        return self.prepare_x(X)

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray = None
    ) -> pd.DataFrame:
        """Fit the model and transform the input data.

        Args:
            X (pd.DataFrame): The input data.
            y (np.ndarray, optional): The target values. Defaults to None.

        Returns:
            pd.DataFrame: The transformed data.

        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target values for the input data.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            np.ndarray: The predicted target values.

        """
        return self.model.predict(self.prepare_x(X))


def get_estimator() -> MyRFR:
    """Get the custom random forest regressor.

    Returns:
        MyRFR: The custom random forest regressor.

    """
    return MyRFR()
