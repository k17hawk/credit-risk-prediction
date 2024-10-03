from scikit_credit_risk.exception import CreditRiskException
import sys
from scikit_credit_risk.logger import logging as  logger
from scikit_credit_risk.entity.config_entity import ModelPusherConfig
from scikit_credit_risk.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from scikit_credit_risk.ml.esitmator import ModelResolver
from scikit_credit_risk.utils import load_object,save_object
import os
from scikit_credit_risk.data_access.model_pusher_artifact import ModelPusherArtifactData


class ModelPusher:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_pusher_config: ModelPusherConfig):
        logger.info(f"{'>>' * 20}Starting Model pusher.{'<<' * 20}")
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_artifact_data = ModelPusherArtifactData()
        self.model_pusher_config = model_pusher_config
        self.model_resolver = ModelResolver(model_dir=self.model_pusher_config.saved_model_dir)

    def push_model(self) -> str:
        try:
            trained_model_path=self.model_trainer_artifact.model_trainer_ref_artifact.trained_model_file_path
            saved_model_path = self.model_resolver.get_save_model_path
            model = load_object(trained_model_path)
            save_object(saved_model_path,obj=model)
            
            return saved_model_path
        except Exception as e:
            raise CreditRiskException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            saved_model_path = self.push_model()
            model_pusher_artifact = ModelPusherArtifact(model_pushed_dir=self.model_pusher_config.pusher_model_dir,
                                    saved_model_dir=saved_model_path)
            logger.info(f"Model pusher artifact: {model_pusher_artifact}")
            self.model_pusher_artifact_data.save_pusher_artifact(model_pusher_artifact = model_pusher_artifact)
            logger.info(f"{'>>' * 20} Model pusher completed.{'<<' * 20}")
            return model_pusher_artifact
        except Exception as e:
            raise CreditRiskException(e, sys)
