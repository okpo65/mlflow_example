import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
import mlflow.pyfunc
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from train import model_type_to_enum
from utils import ModelType


def main(**kwargs):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = kwargs.get('model_type', ModelType.catboost)

    if model_type == ModelType.catboost:
        n_splits = 5
        all_predictions = []
        model_version = 3

        for fold in range(n_splits):
            model_name = f"{model_type}__{fold}"
            model_uri = f"models:/{model_name}/{model_version}"

            model = mlflow.catboost.load_model(model_uri)
            predictions = model.predict(X_test)
            all_predictions.append(predictions)

        y_pred = np.mean(all_predictions, axis=0)
        print(y_pred)
        print(roc_auc_score(y_test, y_pred))



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser(description="Command line arguments with kwargs")

    parser.add_argument('--model_type', type=model_type_to_enum, help='Parameter 1')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
