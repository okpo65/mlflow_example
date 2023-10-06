import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, input_dim),
            nn.Sigmoid()  # MinMax scaling 된 데이터를 재구성하기 위함
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = load_iris()
    X = data.data
    y = data.target

    model_type = "AutoEncoder"

    if model_type == 'catboost':
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            model = CatBoostClassifier(iterations=100, verbose=0)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

            with mlflow.start_run(run_name=f"CatBoost_fold_{fold}"):
                # 모델 저장 및 로깅
                mlflow.catboost.log_model(model, f"catboost_model_fold_{fold}")

                # 필요한 경우, 다른 메트릭이나 파라미터도 함께 로깅 가능
                # 예: mlflow.log_metric(), mlflow.log_param()

                # 각 폴드의 모델 URI 획득
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/catboost_model_fold_{fold}"

                # MLflow 모델 저장소에 모델 등록
                model_name = f"CatBoostModel_fold_{fold}"
                mlflow.register_model(model_uri, model_name)

    elif model_type == 'AutoEncoder':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # PyTorch용 데이터 로더 생성
        tensor_X = torch.Tensor(X_scaled)
        dataset = TensorDataset(tensor_X, tensor_X)  # AutoEncoder는 입력 == 출력이므로
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        autoencoder = SimpleAutoEncoder(input_dim=X_scaled.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

        epochs = 100
        for epoch in range(epochs):
            for batch_x, _ in loader:  # _ 는 사용되지 않는 target입니다.
                outputs = autoencoder(batch_x)
                loss = criterion(outputs, batch_x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        with mlflow.start_run():
            # 모델 저장 및 로깅
            mlflow.pytorch.log_model(autoencoder, "autoencoder_model")

            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/autoencoder_model"

            # MLflow 모델 저장소에 모델 등록
            model_name = "SimpleAutoEncoderModel"
            mlflow.register_model(model_uri, model_name)
