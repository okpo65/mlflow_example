import logging
import sys
import warnings
from enum import Enum

import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from models import CatBoostModel, SimpleAutoEncoder, AutoEncoderModel
from utils.register_mlflow import register_model

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    catboost = 'catboost'
    auto_encoder = 'auto_encoder'


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = load_iris()
    X = data.data
    y = data.target

    model_type = ModelType.catboost
    models = []
    if model_type == ModelType.catboost:
        catboost = CatBoostModel()
        models = catboost.train(X, y, fold=5)

    elif model_type == ModelType.auto_encoder:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        model = SimpleAutoEncoder(input_dim=X_scaled.shape[1])
        autoencoder = AutoEncoderModel(model=model)
        trained_model = autoencoder.train(X=X_scaled)
        models.append(trained_model)

    for fold, model in enumerate(models):
        register_model(model=model, model_type=model_type, uniq='', fold=fold)


