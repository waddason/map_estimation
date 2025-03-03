# AutoEncoder

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

PPG_LENGTH = 1250


class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim, encoding_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim), nn.Sigmoid()
        )
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)

    def fit(self, X, y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        for epoch in range(100):
            self.optimizer.zero_grad()
            outputs = self.autoencoder(X_tensor)
            loss = self.criterion(outputs, X_tensor)
            loss.backward()
            self.optimizer.step()
        return self

    def transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded = self.encoder(X_tensor)
        return encoded.numpy()


def prepare_data_for_autoencoder(X, column_name):
    return np.stack(X[column_name].values)


class SimpleNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 50), nn.ReLU(), nn.Linear(50, 1)
        )
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        for epoch in range(100):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.numpy().flatten()


class PPL:
    def __init__(self):
        self.ppl = Pipeline(
            [
                ("scaler", RobustScaler()),
                (
                    "autoencoder",
                    AutoencoderTransformer(input_dim=20, encoding_dim=10),
                ),
                ("regressor", SimpleNNRegressor(input_dim=10)),
            ]
        )

    def fit(self, X, y):
        self.ppl.fit(X, y)

    def predict(self, X, y=None):
        return self.ppl.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        return mae


def get_estimator():
    model = PPL()
    return model
