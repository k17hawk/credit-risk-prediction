
from .exception import CreditRiskException
import yaml
import os,sys
import pandas as pd
from credit_risk.logger import logging as logger
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

import numpy as np
import dill 

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

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditRiskException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditRiskException(e, sys) from e

def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CreditRiskException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CreditRiskException(e,sys) from e

def get_score(model, X_train, y_train, X_test, y_test, metric_name ) -> float:
    try:
        train_pred = model.predict(X_train)
        
        test_pred = model.predict(X_test)
        metrics = {}
        
        # Select metric based on input
        for metric in metric_name:
            print(metric)
            if metric == 'f1':
                train_f1 = f1_score(y_train, train_pred, average='weighted')
                test_f1 = f1_score(y_test, test_pred, average='weighted')
                metrics['train_f1'] = train_f1
                metrics['test_f1'] = test_f1
                logger.info(f"Train F1 Score: {train_f1}")
                logger.info(f"Test F1 Score: {test_f1}")
            
            elif metric == 'weightedPrecision':
                train_precision = precision_score(y_train, train_pred, average='weighted')
                test_precision = precision_score(y_test, test_pred, average='weighted')
                metrics['train_weightedPrecision'] = train_precision
                metrics['test_weightedPrecision'] = test_precision
                logger.info(f"Train Weighted Precision: {train_precision}")
                logger.info(f"Test Weighted Precision: {test_precision}")
            
            elif metric == 'weightedRecall':
                train_recall = recall_score(y_train, train_pred, average='weighted')
                test_recall = recall_score(y_test, test_pred, average='weighted')
                metrics['train_weightedRecall'] = train_recall
                metrics['test_weightedRecall'] = test_recall
                logger.info(f"Train Weighted Recall: {train_recall}")
                logger.info(f"Test Weighted Recall: {test_recall}")

            elif metric == 'accuracy':
                train_accuracy = accuracy_score(y_train, train_pred)
                test_accuracy = accuracy_score(y_test, test_pred)
                metrics['train_accuracy'] = train_accuracy
                metrics['test_accuracy'] = test_accuracy
                logger.info(f"Train Accuracy: {train_accuracy}")
                logger.info(f"Test Accuracy: {test_accuracy}")

            # Add more metrics as needed based on your requirement
            else:
                logger.error(f"Metric '{metric}' not recognized.")
        
        return metrics

    except Exception as e:
        raise CreditRiskException(e, sys)
