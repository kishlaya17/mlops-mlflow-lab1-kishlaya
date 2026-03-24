# =============================================================================
# MLflow Lab 1 - Experiment Tracking
# =============================================================================
# Original lab used: Wine Quality dataset + ElasticNet model
# Modification:      Diabetes dataset (sklearn) + Ridge Regression model
#
# What this script does:
#   1. Loads the Diabetes dataset from sklearn
#   2. Trains a Ridge Regression model
#   3. Evaluates metrics (RMSE, MAE, R2)
#   4. Logs params, metrics, and model to MLflow
#   5. Stores everything in a local SQLite backend (mlflow.db)
# =============================================================================

import logging
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Use SQLite as the MLflow tracking backend
# This stores all runs, params, metrics in mlflow.db
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def eval_metrics(actual, pred):
    """Calculate RMSE, MAE, and R2 score for model evaluation."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # ------------------------------------------------------------------
    # Load and prepare the Diabetes dataset
    # Features: age, sex, bmi, blood pressure, and 6 serum measurements
    # Target: quantitative measure of disease progression after one year
    # ------------------------------------------------------------------
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    data["target"] = diabetes.target

    # 75% train / 25% test split
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["target"], axis=1)
    test_x = test.drop(["target"], axis=1)
    train_y = train[["target"]]
    test_y = test[["target"]]

    # Hyperparameter for Ridge Regression
    # alpha controls regularization strength (higher = more regularization)
    alpha = 0.5

    # Set the MLflow experiment name
    mlflow.set_experiment("diabetes-ridge-regression")

    # ------------------------------------------------------------------
    # MLflow Run: train, evaluate, and log everything
    # ------------------------------------------------------------------
    with mlflow.start_run() as run:

        # Train Ridge Regression model (modification: was ElasticNet)
        lr = Ridge(alpha=alpha)
        lr.fit(train_x, train_y)

        # Evaluate on test set
        predicted_values = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_values)

        print(f"Ridge Regression model (alpha={alpha}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE:  {mae}")
        print(f"  R2:   {r2}")
        print(f"  Run ID: {run.info.run_id}")

        # Log hyperparameter to MLflow
        mlflow.log_param("alpha", alpha)

        # Log evaluation metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Infer model signature (input/output schema) for safe serving
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        # Log the trained model as an artifact in MLflow
        mlflow.sklearn.log_model(lr, "model", signature=signature)
