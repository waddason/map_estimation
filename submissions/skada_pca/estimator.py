# Skada PCA regressor
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LassoCV

from skada import CORALAdapter

keep_col = ["age", "gender", "ecg", "ppg", "domain"]


class ListToColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, num_columns, chunk_size=1000):
        self.column_name = column_name
        self.num_columns = num_columns
        self.chunk_size = chunk_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Process in chunks to reduce memory usage
        chunks = []
        for start in range(0, len(X), self.chunk_size):
            end = start + self.chunk_size
            chunk = X.iloc[start:end]
            list_df = pd.DataFrame(
                chunk[self.column_name].tolist(), index=chunk.index
            )
            list_df.columns = [
                f"{self.column_name}_{i}" for i in range(self.num_columns)
            ]
            chunk = chunk.drop(columns=[self.column_name])
            chunk = pd.concat([chunk, list_df], axis=1)
            chunks.append(chunk)
        return pd.concat(chunks, axis=0)


class MyEstimator:
    def __init__(self):
        self.pipe = make_pipeline(
            # Keep only 5 columns
            make_column_transformer(
                (SimpleImputer(strategy="median"), ["age"]),
                (OneHotEncoder(), ["gender"]),
                (
                    ListToColumnsTransformer(
                        column_name="ecg", num_columns=1250
                    ),
                    ["ecg"],
                ),
                (
                    ListToColumnsTransformer(
                        column_name="ppg", num_columns=1250
                    ),
                    ["ppg"],
                ),
                # ("passthrough", ["domain"]),
            ),
            IncrementalPCA(n_components=50),
            CORALAdapter(),
            LassoCV(),
        )

    def fit(self, X, y):
        # Transform the domain for use with skada estimator
        X["domain"] = X["domain"].map({"v": 1, "m": -1})
        sample_domain = X["domain"]

        print(f"fit on {X.columns.to_list()} {X.shape}")

        self.pipe.fit(X, y, sample_domain=sample_domain)
        # self.pipe.fit(X, y)
        print("fit done with", self.pipe)
        return self

    def predict(self, X):
        # Issue: missing columns on test data for prediction.
        # Solution: drop bmi/age/weight
        print(f"predict on {X.columns.to_list()} {X.shape}")

        return self.pipe.predict(X)


def get_estimator():
    model = MyEstimator()
    return model
