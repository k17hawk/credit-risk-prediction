from credit_risk.config import mongo_client 
from credit_risk.entity.artifact_entity import DataValidationArtifact

class DataValidationArtifactData:

    def __init__(self):
        self.client = mongo_client
        self.database_name = "credit_artifact"
        self.collection_name = "Data-validation"
        self.collection = self.client[self.database_name][self.collection_name]

    def save_validation_artifact(self, data_valid_artifact: DataValidationArtifact):
        self.collection.insert_one(data_valid_artifact.to_dict())

    def get_valid_artifact(self, query):
        self.collection.find_one(query)