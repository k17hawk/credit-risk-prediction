from scikit_credit_risk.entity.schema import CreditRiskDataSchema
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import logging as  logger
from scikit_credit_risk.entity.artifact_entity import DataTransformationArtifact, \
    PartialModelTrainerMetricArtifact, PartialModelTrainerRefArtifact, ModelTrainerArtifact
from scikit_credit_risk.entity.config_entity import ModelTrainerConfig
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scikit_credit_risk.data_access.model_trainer_artifact import ModelTrainerArtifactData
import os
import sys
from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
import joblib
from scikit_credit_risk.utils import load_numpy_array_data,save_object
import numpy as np


class ModelTrainer:

    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig,
                 schema=CreditRiskDataSchema()
                 ):
        logger.info(f"{'>>' * 20}Starting Model Training.{'<<' * 20}")
        self.model_trainer_artifact_data = ModelTrainerArtifactData()
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.schema = schema
    
    def get_train_test_dataframe(self):
        try:
            
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_file_path = self.data_transformation_artifact.transformed_train_file_path

            train_dataframe: pd.DataFrame = load_numpy_array_data(file_path=train_file_path)
            test_dataframe: pd.DataFrame = load_numpy_array_data(file_path=test_file_path)
           
            return train_dataframe,test_dataframe
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def get_scores(self,model, X_train, y_train, X_test, y_test, metric_name):
        try:
            # Get model predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            metrics = {}

            # Loop through the requested metrics and calculate them
            for metric in metric_name:
                if metric == 'f1':
                    train_f1 = f1_score(y_train, train_pred, average='weighted')
                    test_f1 = f1_score(y_test, test_pred, average='weighted')
                    metrics['train_f1'] = np.float64(train_f1)
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
                    logger.info(f"Metric '{metric}' not recognized.")
            
            return metrics

        except Exception as e:
            logger.info(f"An error occurred: {e}")
            raise CreditRiskException(e, sys)
    
    def get_model(self) -> Pipeline:
        try:
            logger.info("Creating Random Forest Classifier class.")
            
            # Define the RandomForestClassifier step
            random_forest_clf = RandomForestClassifier()
            pipeline = Pipeline(steps=random_forest_clf)
            
            return pipeline
        except Exception as e:
            raise CreditRiskException(e, sys)
    

    def initiate_model_training(self) -> ModelTrainerArtifact:
        transformed_pipeline_file_path = self.data_transformation_artifact.exported_pipeline_file_path
            
        transformed_pipeline = joblib.load(transformed_pipeline_file_path)
        
        try:
            logger.info(f"Loading train and  testing dataset")
            train_array,test_array = self.get_train_test_dataframe()

            logger.info(f"Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]

            logger.info(f"Extracting model config file path")


            model = self.get_model()
            trained_model = model.fit(x_train,y_train)

