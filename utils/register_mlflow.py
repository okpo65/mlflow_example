import mlflow

from train import ModelType


def register_model(model, model_type, uniq='', fold=None):

    run_name = f"{model_type}_{uniq}" if fold is None else f"{model_type}_{uniq}_{fold}"
    with mlflow.start_run(run_name=run_name):

        if model_type == ModelType.catboost:
            mlflow.catboost.log_model(model, run_name)

            # 필요한 경우, 다른 메트릭이나 파라미터도 함께 로깅 가능
            # 예: mlflow.log_metric(), mlflow.log_param()
        elif model_type == ModelType.auto_encoder:
            mlflow.pytorch.log_model(model, run_name)

        # 각 폴드의 모델 URI 획득
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{run_name}"

        # MLflow 모델 저장소에 모델 등록
        model_name = run_name
        mlflow.register_model(model_uri, model_name)
    pass