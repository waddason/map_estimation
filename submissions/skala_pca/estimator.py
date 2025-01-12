# Skala ridge regressor
# Skala_KRR_noWF
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from skada import JDOTRegressor

num_cols = ["age", "height", "weight", "bmi"]
expected_cols = ["age", "height", "weight", "bmi", "domain", "gender"]


class MyEstimator:
    def __init__(self):
        self.clf = make_pipeline(
            make_column_transformer(
                (
                    SimpleImputer(strategy="median"),
                    make_column_selector(
                        pattern=f'^(?:{"|".join(col_name for col_name in ["age", "height", "weight", "bmi"])})$'
                    ),
                ),
                (OneHotEncoder(), ["gender"]),
                ("passthrough", ["domain"]),
            ),
            JDOTRegressor(
                base_estimator=KernelRidge(kernel="rbf", alpha=0.5), alpha=0.01
            ),
        )

    def fit(self, X, y):
        # Transform the domain for use with skada estimator
        X["domain"] = X["domain"].map({"v": 1, "m": -1})
        sample_domain = X["domain"]
        print(f"fit on {X.columns.to_list()} {X.shape}")

        # Save the median value
        self.default_value = {
            col_name: X[col_name].median() for col_name in num_cols
        }
        # Subsample for quicker training
        n_samples_for_train = 15_000
        rng = np.random.default_rng()
        train_idx = rng.choice(
            X.shape[0], size=max(y.shape, n_samples_for_train), replace=False
        )
        X_train = X.iloc[train_idx]  # df
        y_train = y[train_idx]  # np.array
        sample_domain_train = X_train["domain"]
        self.clf.fit(X_train, y_train, sample_domain=sample_domain_train)
        return self

    def predict(self, X):
        # Issue: missing columns on test data for prediction.
        # Solution: replace with median of train data
        print(f"predict on {X.columns.to_list()} {X.shape}")
        for col_name in num_cols:
            if col_name not in X.columns:
                print(f"Col {col_name} missing in X, replace with known median")
                X[col_name] = self.default_value[col_name]

        X["domain"] = X["domain"].map({"v": 1, "m": -1})
        return self.clf.predict(X)


def get_estimator():
    jdot = MyEstimator()
    return jdot
