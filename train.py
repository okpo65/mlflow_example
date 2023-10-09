import argparse
import logging
import warnings

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models import CatBoostModel, SimpleAutoEncoder, AutoEncoderModel
from utils.register_mlflow import ModelType, register_models

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def model_type_to_enum(value):
    try:
        return ModelType(value)
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid model_type: {value}. Expected values are: {list(ModelType)}")


def main(**kwargs):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = kwargs.get('model_type', ModelType.auto_encoder)
    models = []
    if model_type == ModelType.catboost:
        catboost = CatBoostModel()
        models = catboost.train(X_train, y_train, fold=5)
        predictions = catboost.predict(X_test)
        metrics = [roc_auc_score(y_test, y_pred) for y_pred in predictions]

    elif model_type == ModelType.auto_encoder:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = SimpleAutoEncoder(input_dim=X_scaled.shape[1])
        autoencoder = AutoEncoderModel(model=model)
        trained_model = autoencoder.train(X=X_scaled)
        models.append(trained_model)

    register_models(models=models, model_type=model_type, uniq='', metrics=metrics)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser(description="Command line arguments with kwargs")

    parser.add_argument('--model_type', type=model_type_to_enum, help='Parameter 1')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
