from credit_risk.config import mongo_client 
from credit_risk.entity.artifact_entity import ModelTrainerArtifact

class ModelTrainerArtifactData:

    def __init__(self):
        self.client = mongo_client
        self.database_name = "credit_artifact"
        self.collection_name = "model_trainer"
        self.collection = self.client[self.database_name][self.collection_name]

    def save_model_artifact(self, model_trainer_artifact: ModelTrainerArtifact):
        self.collection.insert_one(model_trainer_artifact.to_dict())

    def get_model_artifact(self, query):
        self.collection.find_one(query)