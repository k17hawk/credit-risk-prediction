from scikit_credit_risk.entity.schema import CreditRiskDataSchema
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import logging as  logger
from scikit_credit_risk.entity.artifact_entity import DataTransformationArtifact, \
    PartialModelTrainerMetricArtifact, PartialModelTrainerRefArtifact, ModelTrainerArtifact
from scikit_credit_risk.entity.config_entity import ModelTrainerConfig
from scikit_credit_risk.utils import get_score
from scikit_credit_risk.data_access.model_trainer_artifact import ModelTrainerArtifactData
import os
import sys
from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier 
import joblib

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
            print(f"Train row: {train_dataframe.count()} Test row: {test_dataframe.count()}")
            dataframes: List[pd.DataFrame] = [train_dataframe, test_dataframe]
            return dataframes
        except Exception as e:
            raise CreditRiskException(e, sys)


    def get_scores(self, dataframe: pd.DataFrame, metric_names: List[str]) -> List[tuple]:
        try:
            if metric_names is None:
                metric_names = self.model_trainer_config.metric_list

            scores: List[tuple] = []
            for metric_name in metric_names:
                score = get_score(metric_name=metric_name,
                                  dataframe=dataframe,
                                  label_col=self.schema.target_column,
                                  prediction_col=self.schema.prediction_column_name,)
                scores.append((metric_name, score))
            return scores
        except Exception as e:
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

            train_dataframe_pred = trained_model.predict(X_train)  
            test_dataframe_pred = trained_model.predict(X_test) 
            

            print(f"number of row in training: {train_dataframe_pred.shape}")
            scores = self.get_scores(dataframe=train_dataframe_pred,metric_names=self.model_trainer_config.metric_list)
            train_metric_artifact = PartialModelTrainerMetricArtifact(f1_score=scores[0][1],
                                                                      precision_score=scores[1][1],
                                                                      recall_score=scores[2][1])
            
            logger.info(f"Model trainer train metric: {train_metric_artifact}")


            print(f"number of row in training: {test_dataframe_pred.shape}")
            scores = self.get_scores(dataframe=test_dataframe_pred,metric_names=self.model_trainer_config.metric_list)
            test_metric_artifact = PartialModelTrainerMetricArtifact(f1_score=scores[0][1],
                                                                     precision_score=scores[1][1],
                                                                     recall_score=scores[2][1])
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