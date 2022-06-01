import os
from random import random, randint
import mlflow
import numpy as np
from yaml import load, Loader
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss


def log_mlflow():
    print("Running the test script ...")

    with open("credentials.yml", 'r') as f:
        creds = load(f, Loader)

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = creds['MLFLOW_S3_ENDPOINT_URL']
    os.environ['AWS_ACCESS_KEY_ID'] = creds['AWS_ACCESS_KEY_ID']
    os.environ['AWS_SECRET_ACCESS_KEY'] = creds['AWS_SECRET_ACCESS_KEY']
    os.environ['MLFLOW_TRACKING_USERNAME'] = creds['MLFLOW_TRACKING_USERNAME']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = creds['MLFLOW_TRACKING_PASSWORD']

    # Set the experiment name#Connect to tracking server
    # mlflow.set_tracking_uri("http://10.211.55.5:12346")
    # mlflow.set_tracking_uri("http://0.0.0.0:12346")
    mlflow.set_tracking_uri(creds['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("next_test_log")

    with mlflow.start_run() as run:
        print(mlflow.get_tracking_uri())
        print(mlflow.get_artifact_uri())

        train = np.random.rand(1000, 10)
        test = np.random.rand(100, 10)
        multiplier = np.random.rand(10, 1)
        pre_target = train @ multiplier
        pre_test = test @ multiplier
        target = (pre_target > np.median(pre_target)).astype(int)
        test_target = (pre_test > np.median(pre_test)).astype(int)

        iterations = np.random.randint(20, 150)
        model = CatBoostClassifier(iterations=iterations)
        model.fit(train, target)
        params = model.get_params()

        train_pred = model.predict_proba(train)
        test_pred = model.predict_proba(test)

        stats = {}
        stats['train_AUC'] = roc_auc_score(target, train_pred[:, 1])
        stats['test_AUC'] = roc_auc_score(test_target, test_pred[:, 1])
        stats['train_log_loss'] = log_loss(target, train_pred[:, 1])
        stats['test_log_loss'] = log_loss(test_target, test_pred[:, 1])

        #Test parametes
        mlflow.log_param("iterations", iterations)

        #Test metrics
        mlflow.log_metrics(stats)

        model.save_model("model.cbm")
        mlflow.log_artifact("model.cbm")
        mlflow.catboost.log_model(cb_model=model,
                                  artifact_path="artifact_folder/model.cbm",
                                  registered_model_name="new_test_model")


if __name__ == "__main__":
    log_mlflow()
