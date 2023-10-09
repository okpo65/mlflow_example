from abc import ABCMeta
from typing import Optional, NoReturn

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.models = []
        pass

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame],
               y_valid: Optional[pd.DataFrame]) -> NoReturn:
        raise NotImplementedError

    def train(self, X: pd.DataFrame, y: pd.Series, fold: int):

        if fold == 1:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
            model = self._train(X_train, y_train, X_valid, y_valid)
            self.models.append(model)

        else:
            kf = KFold(n_splits=fold, shuffle=True, random_state=42)

            for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]

                model = self._train(X_train, y_train, X_valid, y_valid)
                self.models.append(model)

        return self.models

    def predict(self, X_test: pd.DataFrame):
        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(X_test)[:, 1])
        return predictions


class CatBoostModel(BaseModel):

    def _train(self,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               X_valid: Optional[pd.DataFrame],
               y_valid: Optional[pd.DataFrame]) -> CatBoostClassifier:
        model = CatBoostClassifier(
            iterations=10000,
            early_stopping_rounds=100,
            random_state=42
        )

        model.fit(
            X=X_train,
            y=y_train,
            eval_set=(X_valid, y_valid),
            verbose=0
        )
        return model
