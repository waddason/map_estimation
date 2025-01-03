# pca
# Dimension reduction of the ecg and ppg signal
# followed by a HistGradientBoosting
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

cat_cols = ["gender"]
# num_cols = ["age", "height", "weight", "bmi"]
num_cols = ["age"]
ts_cols = ["ecg", "ppg", "n_seg"]
train_cols = cat_cols + num_cols + ts_cols

# Remove FutrueWarning for .fillna()
pd.set_option("future.no_silent_downcasting", True)


class MyEstimator:
    def __init__(self, nb_train_samples: int = 100, n_components:int=10):
        self.nb_train_samples = nb_train_samples
        self.nb_components = n_components
        
        # pipeline creation
        self.clf = make_pipeline(
            make_column_transformer(
                
                ("passthrough", self.extract_cols.to_list() + num_cols),
            ),
            HistGradientBoostingRegressor(categorical_features=cat_cols))
            ,
        )

    def _prepare_X(self, X, y=None, train=True):
        """Extract the features of the timeseries signals."""
        # Explode the signals
        X["i_seg"] = X["n_seg"].astype(str) + "_" + X.index.astype(str)
        X_explode = X[["i_seg", "ecg", "ppg"]].explode(
            ["ecg", "ppg"], ignore_index=True
        )
        X_explode["step"] = X_explode.index
        X_explode = X_explode.fillna(0.0)
        # print(X_explode.head())
        # print(X_explode.tail())

        # Ensure ecg and ppg columns are numeric
        X_explode["ecg"] = pd.to_numeric(
            X_explode["ecg"], errors="coerce"
        ).fillna(0.0)
        X_explode["ppg"] = pd.to_numeric(
            X_explode["ppg"], errors="coerce"
        ).fillna(0.0)
        # feature extraction from ecg and ppg
        X_extract = extract_features(
            X_explode,
            column_id="i_seg",
            column_sort="step",
            column_value=None,
            # default_fc_parameters=EfficientFCParameters(),
            default_fc_parameters=MinimalFCParameters(),
        )
        X_extract = X_extract.dropna(axis=1, how="all").fillna(0)
        # Override the index to ensure the merge
        X_extract.index = X.index

        # print(X_explode.head())
        # print(X_explode.tail())
        if train or self.extract_cols is None:
            assert (
                y is not None
            ), "You must provide y_train for feature selection."
            X_extract = select_features(X_extract, y)
            self.extract_cols = X_extract.columns
        else:
            # Use the same columns as in training
            X_extract = X_extract[self.extract_cols]

        # Merge back into the dataframe
        X_train = pd.concat([X, X_extract], axis=1)
        return X_train

    def fit(self, X, y):
        """Fit the estimator."""
        rng = np.random.default_rng()
        self.nb_train_samples = min(self.nb_train_samples, len(y))
        train_idx = rng.choice(
            X.shape[0], size=min(len(y), self.nb_train_samples), replace=False
        )
        # select only the needed rows for training
        y_train = y[train_idx]
        X_train = X[train_cols].iloc[train_idx]
        X_train = self._prepare_X(X_train, y_train)
        print(
            f"Fit on {X.shape=} {X_train.shape=} {y_train.shape=} {self.nb_train_samples=}"
        )
        # print(f"fit on {X_train.columns.to_list()} {X_train.shape}")



        self.clf.fit(
            X_train,
            y_train,
        )
        return self

    def predict(self, X):
        X_pred = self._prepare_X(X, train=False)
        # Issue: missing columns on test data for prediction.
        # Solution: remove columns from training
        print(f"predict on {X_pred.shape=}")

        return self.clf.predict(X_pred)


def get_estimator():
    model = MyEstimator(nb_train_samples=200)
    return model
