from scikit_credit_risk.entity.artifact_entity import (ModelEvaluationArtifact, DataValidationArtifact,
    ModelTrainerArtifact)
from scikit_credit_risk.entity.config_entity import ModelEvaluationConfig
from scikit_credit_risk.entity.schema import CreditRiskDataSchema
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import logging as logger
import sys

from scikit_credit_risk.utils import get_score

from scikit_credit_risk.data_access.model_eval_artifcat import ModelEvaluationArtifactData
from scikit_credit_risk.ml.esitmator import  ModelResolver,CreditRiskEstimator
import pandas as pd
import joblib

class ModelEvaluation:

    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_eval_config: ModelEvaluationConfig,
                 schema=CreditRiskDataSchema()
                 ):
        try:
            self.model_eval_artifact_data = ModelEvaluationArtifactData()
            self.data_validation_artifact = data_validation_artifact
            self.model_eval_config = model_eval_config
            self.model_trainer_artifact = model_trainer_artifact
            self.schema = schema
            self.model_resolver = ModelResolver()
            self.credit_estimator = CreditRiskEstimator()
        except Exception as e:
            raise CreditRiskException(e, sys)

    def read_data(self) -> pd.DataFrame:
        try:
            file_path = self.data_validation_artifact.accepted_file_path
            dataframe: pd.DataFrame = pd.read_parquet(file_path,engine="pyarrow")
            return dataframe
        except Exception as e:
            # Raising an exception.
            raise CreditRiskException(e, sys)


    def evaluate_trained_model(self) -> ModelEvaluationArtifact:
        try:
            if not self.model_resolver.is_model_present:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    model_accepted=True,
                    changed_accuracy= None,
                    trained_model_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path,
                    best_model_path = None,
                    active=True
                )
                return  model_evaluation_artifact

            #set initial flag
            is_model_accepted, is_active = False, False


            #obtain required directory path
            trained_model_file_path = self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
            trained_model = joblib.load(trained_model_file_path)



            dataframe: pd.DataFrame = self.read_data()


            print("applying pipeline to data")
            best_model_path = self.model_resolver.get_best_model_path() 

            best_model_dataframe = self.credit_estimator.transform(dataframe)
            
    
            print("applying pipeline completed..")


            print("applying pipeline from best model")
            trained_model_dataframe = trained_model.transform(dataframe)

            print("pipeline executed successfully...")

     
            trained_model_f1_score = get_score(dataframe=trained_model_dataframe, metric_name="f1",
                                            label_col=self.schema.target_column,
                                            prediction_col=self.schema.prediction_column_name)
            #compute f1 score for best model
            best_model_f1_score = get_score(dataframe=best_model_dataframe, metric_name="f1",
                                            label_col=self.schema.target_column,
                                            prediction_col=self.schema.prediction_column_name)
            
            print(f"Trained_model_f1_score: {trained_model_f1_score}, Best model f1 score: {best_model_f1_score}")
            print("no error")

            logger.info(f"Trained_model_f1_score: {trained_model_f1_score}, Best model f1 score: {best_model_f1_score}")
            #improved accuracy
            changed_accuracy = trained_model_f1_score - best_model_f1_score

            
            if changed_accuracy >= self.model_eval_config.threshold:
                is_model_accepted, is_active = True, True
            model_evaluation_artifact = ModelEvaluationArtifact(model_accepted=is_model_accepted,
                                                                changed_accuracy=changed_accuracy,
                                                                trained_model_path=trained_model_file_path,
                                                                best_model_path=best_model_path,
                                                                active=is_active
                                                                )
            return model_evaluation_artifact
        except Exception as e:
            raise CreditRiskException(e,sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logger.info(f"{'>>' * 20}Starting model Evaluation.{'<<' * 20}")
            model_accepted = True
            is_active = True
            model_evaluation_artifact = self.evaluate_trained_model()
            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            self.model_eval_artifact_data.save_eval_artifact(model_eval_artifact=model_evaluation_artifact)
            logger.info(f"{'>>' * 20}model Evaluation completed...{'<<' * 20}")
            return model_evaluation_artifact
        
        except Exception as e:
            raise CreditRiskException(e, sys)