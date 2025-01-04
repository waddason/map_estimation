# Use of neurokit feature extraction
import neurokit2 as nk
import numpy as np
import pandas as pd
from time import time
import scipy.stats as stats
from sklearn.compose import make_column_transformer

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


###############################################################################
# Constants
###############################################################################
SAMPLING_RATE = 125

ECG_DELTAS = {  # name : [end, start]
    "PR": ["ECG_R_Offsets", "ECG_P_Onsets"],
    "QS": ["ECG_S_Peaks", "ECG_Q_Peaks"],
    "RT": ["ECG_T_Offsets", "ECG_R_Onsets"],
    "TP": ["ECG_P_Offsets", "ECG_T_Onsets"],
    "PP": ["ECG_P_Peaks", "ECG_P_Onsets"],
    "TT": ["ECG_T_Peaks", "ECG_T_Onsets"],
}

ppg_analys_cols = [
    "PPG_Rate_Mean",
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_SDANN1",
    "HRV_SDNNI1",
    "HRV_SDANN2",
    "HRV_SDNNI2",
    "HRV_SDANN5",
    "HRV_SDNNI5",
    "HRV_RMSSD",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_MedianNN",
    "HRV_MadNN",
    "HRV_MCVNN",
    "HRV_IQRNN",
    "HRV_SDRMSSD",
    "HRV_Prc20NN",
    "HRV_Prc80NN",
    "HRV_pNN50",
    "HRV_pNN20",
    "HRV_MinNN",
    "HRV_MaxNN",
    "HRV_HTI",
    "HRV_TINN",
    "HRV_ULF",
    "HRV_VLF",
    "HRV_LF",
    "HRV_HF",
    "HRV_VHF",
    "HRV_TP",
    "HRV_LFHF",
    "HRV_LFn",
    "HRV_HFn",
    "HRV_LnHF",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",
    "HRV_CSI",
    "HRV_CVI",
    "HRV_CSI_Modified",
    "HRV_PIP",
    "HRV_IALS",
    "HRV_PSS",
    "HRV_PAS",
    "HRV_GI",
    "HRV_SI",
    "HRV_AI",
    "HRV_PI",
    "HRV_C1d",
    "HRV_C1a",
    "HRV_SD1d",
    "HRV_SD1a",
    "HRV_C2d",
    "HRV_C2a",
    "HRV_SD2d",
    "HRV_SD2a",
    "HRV_Cd",
    "HRV_Ca",
    "HRV_SDNNd",
    "HRV_SDNNa",
    "HRV_DFA_alpha1",
    "HRV_MFDFA_alpha1_Width",
    "HRV_MFDFA_alpha1_Peak",
    "HRV_MFDFA_alpha1_Mean",
    "HRV_MFDFA_alpha1_Max",
    "HRV_MFDFA_alpha1_Delta",
    "HRV_MFDFA_alpha1_Asymmetry",
    "HRV_MFDFA_alpha1_Fluctuation",
    "HRV_MFDFA_alpha1_Increment",
    "HRV_ApEn",
    "HRV_SampEn",
    "HRV_ShanEn",
    "HRV_FuzzyEn",
    "HRV_MSEn",
    "HRV_CMSEn",
    "HRV_RCMSEn",
    "HRV_CD",
    "HRV_HFD",
    "HRV_KFD",
    "HRV_LZC",
]

ppg_nan_cols = [
    "HRV_SDANN1",
    "HRV_SDNNI1",
    "HRV_SDANN2",
    "HRV_SDNNI2",
    "HRV_SDANN5",
    "HRV_SDNNI5",
    "HRV_ULF",
    "HRV_VLF",
    "HRV_LF",
    "HRV_HF",
    "HRV_VHF",
    "HRV_LFHF",
    "HRV_LFn",
    "HRV_HFn",
    "HRV_LnHF",
    "HRV_MSEn",
    "HRV_CMSEn",
    "HRV_RCMSEn",
    "HRV_HFD",
]

ppg_keep_cols = [
    col_name for col_name in ppg_analys_cols if col_name not in ppg_nan_cols
]


###############################################################################
# ECG preprocessing
###############################################################################
def get_waves_len(waves_peak) -> dict[str : np.float64]:
    """Return the Duration of events in ECG with mean and std.
    - P, R, T waves.
    - P-R, R-T, T-P, Q-S deltas
    - heart rate"""
    val_dict = {}
    # Phase duration
    for wave_n in ["P", "R", "T"]:
        waves_len = np.array(waves_peak[f"ECG_{wave_n}_Offsets"]) - np.array(
            waves_peak[f"ECG_{wave_n}_Onsets"]
        )
        waves_len = waves_len[~np.isnan(waves_len)]
        if waves_len.size > 0:
            val_dict[f"ECG_{wave_n}_Duration_mean"] = waves_len.mean()
            val_dict[f"ECG_{wave_n}_Duration_std"] = waves_len.std()
        else:
            val_dict[f"ECG_{wave_n}_Duration_mean"] = 0.0
            val_dict[f"ECG_{wave_n}_Duration_std"] = 1.0

    # Delta between phases PR, peak QS, RT, TP

    for delta_name, [end, start] in ECG_DELTAS.items():
        delta_len = np.subtract(
            *np.broadcast_arrays(
                np.array(waves_peak[end]), np.array(waves_peak[start])
            )
        )
        min_len = min(len(waves_peak[end]), len(waves_peak[start]))
        delta_len = delta_len[:min_len]
        delta_len = delta_len[~np.isnan(delta_len)]
        if delta_len.size > 0:
            val_dict[f"ECG_{delta_name}_delta_mean"] = delta_len.mean()
            val_dict[f"ECG_{delta_name}_delta_std"] = delta_len.std()
        else:
            val_dict[f"ECG_{delta_name}_delta_mean"] = 0.0
            val_dict[f"ECG_{delta_name}_delta_std"] = 1.0

    # Heart rate
    # Should have use the R_peaks, but it is not computed in waves_peak, so use
    # Q_peaks instead.
    dfeet = np.diff(waves_peak["ECG_Q_Peaks"])
    dfeet = dfeet[~np.isnan(dfeet)]
    if dfeet.size > 0:
        # Result in beats per minute
        dfeet = 60 / dfeet * SAMPLING_RATE
        val_dict["ECG_Heartrate_mean"] = dfeet.mean()
        val_dict["ECG_Heartrate_std"] = dfeet.std()
    else:
        val_dict["ECG_Heartrate_mean"] = 0.0
        val_dict["ECG_Heartrate_std"] = 1.0

    return val_dict


def get_peak_stat_values(one_ecg_as_list, waves_peak):
    val_dict = {}
    for peak, idx in waves_peak.items():
        idx = [x for x in idx if not np.isnan(x)]
        idx = np.array(idx, dtype=int)
        values = one_ecg_as_list[idx]
        if values.size > 0:
            val_dict[f"{peak}_val_mean"] = values.mean()
            val_dict[f"{peak}_val_std"] = values.std()
        else:
            val_dict[f"{peak}_val_mean"] = values.mean()
            val_dict[f"{peak}_val_std"] = values.std()
        # print(peak, ":", values.mean(), ",", values.std())
    return val_dict


def extract_ecg_features(one_ecg_as_list) -> pd.Series:
    """Extract relevant features from the ecg."""

    _, waves_peak = nk.ecg_delineate(
        one_ecg_as_list, sampling_rate=SAMPLING_RATE
    )
    features = {
        **get_waves_len(waves_peak),
        **get_peak_stat_values(one_ecg_as_list, waves_peak),
    }

    return pd.Series(features)


###############################################################################
# PPG Preprocessing
###############################################################################
def extract_ppg_features(one_ppg_as_list) -> pd.Series:
    """Extract relevant features from the ppg"""
    signal, info = nk.ppg_process(one_ppg_as_list, sampling_rate=SAMPLING_RATE)
    ppg_ana = nk.ppg_analyze(signal, SAMPLING_RATE)
    ppg_ana = ppg_ana[ppg_keep_cols].fillna(0.0)
    return ppg_ana.iloc[0]  # force return of Series


def safe_extract_ppg_features(one_ppg_as_list) -> pd.Series:
    """Extract relevant features from the ppg, return Series full of 0.0 if specific ValueError occurs."""
    try:
        return extract_ppg_features(one_ppg_as_list)
    except ValueError as e:
        if "NeuroKit error: the window" in str(e):
            return pd.Series(0.0, index=ppg_keep_cols)
        else:
            raise e


###############################################################################
# Estimator
###############################################################################
# Random Search cross validation CLF with simple Lasso
# Lasso parameters
param_lasso = {
    "alpha": stats.loguniform(1e-2, 1e0),
    "tol": stats.loguniform(1e-5, 1e-1),
}
# run randomized search
n_iter_search = 15


class MyEstimator:
    def __init__(self):
        # pipeline creation
        self.clf = make_pipeline(
            make_column_transformer(
                (
                    FunctionTransformer(
                        lambda x: x.apply(extract_ecg_features)
                    ),
                    "ecg",
                ),
                # (
                #     FunctionTransformer(
                #         lambda x: x.apply(safe_extract_ppg_features)
                #     ),
                #     "ppg",
                # ),
                (SimpleImputer(strategy="median"), ["age"]),
                (OneHotEncoder(), ["gender", "domain"]),
                # ("passthrough", ["domain"]),
            ),
            RandomizedSearchCV(
                Lasso(), param_distributions=param_lasso, n_iter=n_iter_search
            ),
        )

    def fit(self, X, y):
        """Fit the estimator."""
        start = time()
        print(f"Start fit on {X.shape=} {y.shape=}")

        self.clf.fit(
            X,
            y,
        )
        elapsed_time = time() - start
        minutes, seconds = divmod(elapsed_time, 60)
        print(
            f"End fit for {int(minutes)}min {seconds:.2f}s for {self.clf[-1]}"
        )
        return self

    def predict(self, X):
        print(f"predict on {X.shape=}")

        return self.clf.predict(X)


def get_estimator():
    model = MyEstimator()
    return model
