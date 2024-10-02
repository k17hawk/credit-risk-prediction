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

    def get_train_test_dataframe(self) -> List[pd.DataFrame]:
        try:
            
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_dataframe: pd.DataFrame = pd.read_parquet(train_file_path)
            test_dataframe: pd.DataFrame = pd.read_parquet(test_file_path)
            dataframes: List[pd.DataFrame] = [train_dataframe, test_dataframe]
            return dataframes
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
            steps = []
            logger.info("Creating Random Forest Classifier class.")
            
            # Define the RandomForestClassifier step
            random_forest_clf = RandomForestClassifier()
            steps.append(('random_forest', random_forest_clf))  
            
            # Create the pipeline with the defined steps
            pipeline = Pipeline(steps=steps)
            
            return pipeline
        except Exception as e:
            raise CreditRiskException(e, sys)



    def export_trained_model(self, model) -> PartialModelTrainerRefArtifact:
        try:

            transformed_pipeline_file_path = self.data_transformation_artifact.exported_pipeline_file_path
            
            transformed_pipeline = joblib.load(transformed_pipeline_file_path)

            updated_steps = transformed_pipeline.steps + model.steps
            transformed_pipeline.steps = updated_steps
     
            logger.info("Creating trained model directory")
            trained_model_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)

            trained_model_file_path = os.path.join(trained_model_path,'trained_model.pkl')
            joblib.dump(transformed_pipeline,trained_model_file_path)


            ref_artifact = PartialModelTrainerRefArtifact(
                trained_model_file_path=trained_model_file_path,
                )

            logger.info(f"Model trainer reference artifact: {ref_artifact}")
            return ref_artifact

        except Exception as e:
            raise CreditRiskException(e, sys)


    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            dataframes = self.get_train_test_dataframe()

            train_dataframe, test_dataframe = dataframes[0], dataframes[1]

            X_train = train_dataframe.drop([self.schema.target_column], axis=1)
            y_train = train_dataframe[self.schema.target_column]

            X_test = test_dataframe.drop([self.schema.target_column], axis=1)
            y_test = test_dataframe[self.schema.target_column]

            model = self.get_model()
            trained_model = model.fit(X_train,y_train)
            scores = self.get_scores(model, X_train, y_train, X_test, y_test,metric_name=self.model_trainer_config.metric_list)

            train_f1 = scores.get('train_f1', None)
            test_f1 = scores.get('test_f1', None)
            train_weightedPrecision = scores.get('train_weightedPrecision', None)
            test_weightedPrecision = scores.get('test_weightedPrecision', None)
            train_weightedRecall = scores.get('train_weightedRecall', None)
            test_weightedRecall = scores.get('test_weightedRecall', None)
            train_accuracy = scores.get('train_accuracy', None)
            test_accuracy = scores.get('test_accuracy', None)

           
            train_metric_artifact = PartialModelTrainerMetricArtifact(f1_score=train_f1,
                                                                      precision_score=train_weightedPrecision,
                                                                      recall_score=train_weightedRecall)
            
            logger.info(f"Model trainer train metric: {train_metric_artifact}")



            test_metric_artifact = PartialModelTrainerMetricArtifact(f1_score=test_f1,
                                                                     precision_score=test_weightedPrecision,
                                                                     recall_score=test_weightedRecall)
            logger.info(f"Model trainer test metric: {test_metric_artifact}")
            
            ref_artifact = self.export_trained_model(model=trained_model)
            model_trainer_artifact = ModelTrainerArtifact(model_trainer_ref_artifact=ref_artifact,
                                                          model_trainer_train_metric_artifact=train_metric_artifact,
                                                          model_trainer_test_metric_artifact=test_metric_artifact)
            
            self.model_trainer_artifact_data.save_model_artifact(model_trainer_artifact=model_trainer_artifact)

            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            logger.info(f"{'>>' * 20}Model Training End {'<<' * 20}")
            

            return model_trainer_artifact

        except Exception as e:
            raise CreditRiskException(e, sys)