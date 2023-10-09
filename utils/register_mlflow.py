from enum import Enum

import mlflow


class ModelType(Enum):
    catboost = 'catboost'
    auto_encoder = 'auto_encoder'


def register_models(models, model_type, uniq='', metrics=[]):


    for fold, model in enumerate(models):
        run_name = f"{model_type}_{uniq}" if fold is None else f"{model_type}_{uniq}_{fold}"
        with mlflow.start_run(run_name=run_name):

            if model_type == ModelType.catboost:
                mlflow.catboost.log_model(model, run_name)
                mlflow.log_metric(key="roc", value=metrics[fold])

                params = model.get_params()
                for key, value in params.items():
                    mlflow.log_param(key=key, value=value)

            elif model_type == ModelType.auto_encoder:
                mlflow.pytorch.log_model(model, run_name)

            # 각 폴드의 모델 URI 획득
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/{run_name}"

            # MLflow 모델 저장소에 모델 등록
            model_name = run_name
            mlflow.register_model(model_uri, model_name)