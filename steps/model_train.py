import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    params: ModelNameConfig,
) -> RegressorMixin:
    """
    ZenML step to train a regression model based on provided parameters.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.
        params (ModelNameConfig): Configuration parameters including model_name.
    Returns:
        RegressorMixin: The trained regression model.
    """
    try:
        if params.model_name == "Linear Regression":
            trained_model = LinearRegressionModel().train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model '{params.model_name}' is not supported.")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e

