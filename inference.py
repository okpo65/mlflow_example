import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import mlflow.pyfunc
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":


    data = load_iris()
    X = data.data
    y = data.target

    model_type = 'catboost'

    if model_type == 'catboost':
        n_splits = 5
        all_predictions = []
        model_version = 1

        for fold in range(n_splits):
            model_name = f"CatBoostModel_fold_{fold}"
            # 최신 버전의 모델을 불러오기 위한 URI (다른 버전이 필요하다면 URI를 변경해야 합니다)
            model_uri = f"models:/{model_name}/{model_version}"

            model = mlflow.catboost.load_model(model_uri)
            predictions = model.predict(X)
            all_predictions.append(predictions)

        print(np.mean(all_predictions, axis=0))

    elif model_type == 'autoencoder':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model_name = "SimpleAutoEncoderModel"
        model_version = 1

        model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
        with torch.no_grad():
            sample_data = torch.Tensor(X_scaled[:5])  # 예시로 처음 5개의 데이터를 사용
            reconstructed_data = model(sample_data).numpy()

        print("Original Data:")
        print(X_scaled[:5])
        print("\nReconstructed Data:")
        print(reconstructed_data)