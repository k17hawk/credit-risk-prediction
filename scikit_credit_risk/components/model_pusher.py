"""
author @ kumar dahaltest_precision
this code is written to push the model in the cloud
"""

from scikit_credit_risk import logging
from scikit_credit_risk.exception import CreditException
from scikit_credit_risk.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact 
from scikit_credit_risk.entity.config_entity import ModelPusherConfig
import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact

        except Exception as e:
            raise CreditException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            #get the model path or trained or old model from artifact
            evaluated_model_file_path = self.model_evaluation_artifact.evaluated_model_path

            #directory to export model
            export_dir = self.model_pusher_config.export_dir_path
            #get the file name from evaluated_model_file_path
            model_file_name = os.path.basename(evaluated_model_file_path)
            #create the directory to copy the model 
            export_model_file_path = os.path.join(export_dir, model_file_name)
            logging.info(f"Exporting model file: [{export_model_file_path}]")
            os.makedirs(export_dir, exist_ok=True)

            #copy from source to destination
            shutil.copy(src=evaluated_model_file_path, dst=export_model_file_path)
            #we can call a function to save model in cloud from here create the save_models folder in cloud
            
            logging.info(
                f"Trained model: {evaluated_model_file_path} is copied in export dir:[{export_model_file_path}]")

            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_model_file_path
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise CreditException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise CreditException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")