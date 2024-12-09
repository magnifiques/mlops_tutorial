import logging
import pandas as pd
from zenml import step
from src.model_development import LinearRegression
from sklearn.base import RegressorMixin
from config import ModelNameConfig

@step
def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, config: ModelNameConfig) -> RegressorMixin:
    """Trains the model on the Ingested data

    Args:
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.DataFrame, 
        y_test: pd.DataFrame,
        config: ModelNameConfig
    """
    try:
        model = None
        
        if config.model_name == "LinearRegression":
            model = LinearRegression()
            trained_model = model.train(X_train, y_train)
            
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training mode: {}".format(e))
        raise e