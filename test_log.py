import os
from random import random, randint
import mlflow
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss


def log_mlflow():
    print("Running the test script ...")

    # Set the experiment name#Connect to tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:12346")

    mlflow.set_experiment("test_log")
    mlflow.start_run()

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
    mlflow.log_param("iterations", params['iterations'])

    #Test metrics
    mlflow.log_metrics(stats)

    mlflow.catboost.log_model(model, "test_model")
    mlflow.end_run()

if __name__ == "__main__":
    log_mlflow()
