
from .exception import CreditRiskException
import yaml
import os,sys
import pandas as pd
from credit_risk.logger import logging as logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

def write_yaml_file(file_path: str, data: dict = None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as e:
        raise CreditRiskException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditRiskException(e, sys) from e

def get_score(dataframe: pd.DataFrame, metric_name, label_col, prediction_col) -> float:
    try:
        y_true = dataframe[label_col]  # Ground truth labels
        y_pred = dataframe[prediction_col]  # Predicted labels
        
        # Select metric based on input
        if metric_name == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric_name == 'f1':
            score = f1_score(y_true, y_pred, average='weighted')
        elif metric_name == 'precision':
            score = precision_score(y_true, y_pred, average='weighted')
        elif metric_name == 'recall':
            score = recall_score(y_true, y_pred, average='weighted')
        elif metric_name == 'log_loss':
            score = log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        print(f"{metric_name} score: {score}")
        logger.info(f"{metric_name} score: {score}")
        return score
    except Exception as e:
        raise CreditRiskException(e, sys)
