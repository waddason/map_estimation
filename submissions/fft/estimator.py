# This estimator focuses on the data from the PPG
# In the spirit of non-invasive method to get the map
# drop the ECG and possible empty columns
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

keep_col = ["age", "gender", "ppg", "ecg"]


class MyHGBRNoPPL(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins: int = 30, random_state: int = 1) -> None:
        """Create a custom random forest regressor for MAP estimation."""
        self.max_bins = max_bins
        self.random_state = random_state
        # Filtering param
        self.cutoff = 5.0  # Cutoff frequency in Hz
        self.fs = 125.0  # Sampling frequency in Hz in input data
        self.order = 5
        self.model = HistGradientBoostingRegressor(
            max_bins=self.max_bins,
            random_state=self.random_state,
            loss="absolute_error",
            categorical_features=["gender"],
        )
        self.transform = self.prepare_x

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MyHGBR":
        # Check for duplicate columns
        print(f"Call fit on {X.shape[0]}, {X.columns}")
        X = X[y != -1]
        y = y[y != -1]
        prepared_x = self.transform(X)
        self.model.fit(prepared_x, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        print(f"Call predict on {X.shape[0]}, {X.columns}")
        prepared_x = self.transform(X)
        return self.model.predict(prepared_x)

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

    def compute_pulse_area(self, ppg_signal: np.ndarray[float]) -> float:
        """Compute the pulse area from PPG signal.

        Args:
            ppg_signal (np.ndarray[float]): The PPG signal.

        Returns:
            float: The computed pulse area.

        """
        return np.trapz(ppg_signal)

    def compute_ppg_features(self, ppg_signal: np.ndarray[float]) -> dict:
        """Compute various features from PPG signal.

        Args:
            ppg_signal (np.ndarray[float]): The PPG signal.

        Returns:
            dict: A dictionary containing the computed features.

        """
        ppg_peaks, _ = find_peaks(ppg_signal, distance=self.fs / 2)
        rr_intervals = np.diff(ppg_peaks) / self.fs

        features = {
            "ptt": np.nanmean(rr_intervals) if len(ppg_peaks) > 1 else np.nan,
            "prv": np.nanstd(rr_intervals) if len(rr_intervals) > 1 else np.nan,
            "pulse_amplitude": np.nanmax(ppg_signal) - np.nanmin(ppg_signal),
            "pulse_width": np.nanmean(rr_intervals)
            if len(ppg_peaks) > 1
            else np.nan,
            "pulse_area": np.trapz(ppg_signal),
            "dicrotic_notch": np.nanmean(rr_intervals)
            if len(ppg_peaks) > 1
            else np.nan,
        }

        return features

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
        prepared_x["PPG_filtered"] = sosfiltfilt(sos, x["ppg"])
        prepared_x["ECG_filtered"] = sosfiltfilt(sos, x["ecg"])

        # Heart rate from ECG
        prepared_x["heart_rate"] = prepared_x["ECG_filtered"].apply(
            self.compute_heart_rate
        )

        # Additional PPG features
        prepared_x["pulse_area"] = prepared_x["PPG_filtered"].apply(
            self.compute_pulse_area
        )

        # Compute PPG features and merge them into prepared_x
        ppg_features = prepared_x["PPG_filtered"].apply(
            self.compute_ppg_features
        )
        ppg_features_df = pd.json_normalize(ppg_features)
        ppg_features_df.index = prepared_x.index
        prepared_x = pd.concat([prepared_x, ppg_features_df], axis=1)
        prepared_x = prepared_x.drop(columns=["ECG_filtered", "PPG_filtered"])
        print(
            f"Prepared {prepared_x.shape[0]} rows, {list(prepared_x.columns)}",
        )

        return prepared_x


class MyHGBR(BaseEstimator, TransformerMixin):
    """Custom random forest."""

    def __init__(self, max_bins: int = 30, random_state: int = 1) -> None:
        """Create a custom random forest regressor for MAP estimation."""
        self.max_bins = max_bins
        self.random_state = random_state
        # Filtering param
        self.cutoff = 5.0  # Cutoff frequency in Hz
        self.fs = 125.0  # Sampling frequency in Hz in input data
        self.order = 5
        self.model = make_pipeline(
            FunctionTransformer(self.prepare_x),
            make_column_transformer(
                (
                    OneHotEncoder(
                        categories=[["M", "F"]],
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                    ["gender"],
                ),
                remainder="passthrough",
            ),
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_bins=self.max_bins,
                random_state=self.random_state,
                loss="absolute_error",
            ),
        )

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

    def compute_pulse_area(self, ppg_signal: np.ndarray[float]) -> float:
        """Compute the pulse area from PPG signal.

        Args:
            ppg_signal (np.ndarray[float]): The PPG signal.

        Returns:
            float: The computed pulse area.

        """
        return np.trapz(ppg_signal)

    def compute_ppg_features(self, ppg_signal: np.ndarray[float]) -> dict:
        """Compute various features from PPG signal.

        Args:
            ppg_signal (np.ndarray[float]): The PPG signal.

        Returns:
            dict: A dictionary containing the computed features.

        """
        ppg_peaks, _ = find_peaks(ppg_signal, distance=self.fs / 2)
        rr_intervals = np.diff(ppg_peaks) / self.fs

        features = {
            "ptt": np.nanmean(rr_intervals) if len(ppg_peaks) > 1 else np.nan,
            "prv": np.nanstd(rr_intervals) if len(rr_intervals) > 1 else np.nan,
            "pulse_amplitude": np.nanmax(ppg_signal) - np.nanmin(ppg_signal),
            "pulse_width": np.nanmean(rr_intervals)
            if len(ppg_peaks) > 1
            else np.nan,
            "pulse_area": np.trapz(ppg_signal),
            "dicrotic_notch": np.nanmean(rr_intervals)
            if len(ppg_peaks) > 1
            else np.nan,
        }

        return features

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
        prepared_x["PPG_filtered"] = sosfiltfilt(sos, x["ppg"])
        prepared_x["ECG_filtered"] = sosfiltfilt(sos, x["ecg"])

        # Heart rate from ECG
        prepared_x["heart_rate"] = prepared_x["ECG_filtered"].apply(
            self.compute_heart_rate
        )

        # Additional PPG features
        prepared_x["pulse_area"] = prepared_x["PPG_filtered"].apply(
            self.compute_pulse_area
        )

        # Compute PPG features and merge them into prepared_x
        ppg_features = prepared_x["PPG_filtered"].apply(
            self.compute_ppg_features
        )
        ppg_features_df = pd.json_normalize(ppg_features)
        ppg_features_df.index = prepared_x.index
        prepared_x = pd.concat([prepared_x, ppg_features_df], axis=1)
        prepared_x = prepared_x.drop(columns=["ECG_filtered", "PPG_filtered"])
        print(
            f"Prepared {prepared_x.shape[0]} rows, {list(prepared_x.columns)}",
        )

        return prepared_x

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MyHGBR":
        """Fit the model to the data.

        Args:
            X (pd.DataFrame): The input data.
            y (np.ndarray): The target values.

        Returns:
            MyHGB: The fitted model.

        """
        # Check for duplicate columns
        print(f"Call fit on {X.shape[0]}, {X.columns}")
        X = X[y != -1]
        y = y[y != -1]
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target values for the input data.

        Args:
            X (pd.DataFrame): The input data.

        Returns:
            np.ndarray: The predicted target values.

        """
        print(f"Call predict on {X.shape[0]}, {X.columns}")
        return self.model.predict(X)


class MyHGBR2:
    """Custom random forest."""

    def __init__(self, max_bins: int = 30, random_state: int = 1) -> None:
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
                remainder="passthrough",
            ),
            HistGradientBoostingRegressor(
                max_bins=max_bins, random_state=random_state
            ),
        )
        # Filtering param
        self.cutoff = 5.0  # Cutoff frequency in Hz
        self.fs = 125.0  # Sampling frequency in Hz in input data
        self.order = 5

    def compute_heart_rate(self, ecg_signal: np.ndarray[float]) -> float:
        """Compute the heart rate from ecg."""
        peaks, _ = find_peaks(ecg_signal, distance=self.fs / 2)
        rr_intervals = np.diff(peaks) / self.fs
        # Heart rate
        return (
            60 / np.nanmean(rr_intervals) if len(rr_intervals) > 0 else np.nan
        )

    def compute_ptt(self, ppg_signal: np.ndarray[float]) -> float:
        """Compute the pulste transit time from ptt."""
        ppg_peaks, _ = find_peaks(ppg_signal, distance=self.fs / 2)
        # PPT
        return (
            np.nanmean(np.diff(ppg_peaks) / self.fs)
            if len(ppg_peaks) > 1
            else np.nan
        )

    def prepare_x(self, x: pd.DataFrame) -> pd.DataFrame:
        """Use filtering techniques to remove noise from the PPG and ECG."""
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
        assert "ppg" in x.columns, "Error, no ppg column, in input data"
        prepared_x["PPG_filtered"] = sosfiltfilt(sos, x["ppg"])
        assert "ecg" in x.columns, "Error, no ecg column, in input data"
        prepared_x["ECG_filtered"] = sosfiltfilt(sos, x["ecg"])

        # Heart rate from ECG
        prepared_x["heart_rate"] = prepared_x["ECG_filtered"].apply(
            self.compute_heart_rate
        )

        # Pulse transit time (PTT)
        prepared_x["ptt"] = prepared_x["PPG_filtered"].apply(self.compute_ptt)

        return prepared_x.drop(columns=["ECG_filtered", "PPG_filtered"])

    def fit(self, X, y):
        return self.model.fit(self.prepare_x(X), y)

    def predict(self, X):
        return self.model.predict(self.prepare_x(X))


def get_estimator():
    model = MyHGBRNoPPL()
    return model
