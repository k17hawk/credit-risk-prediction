from credit_risk.config import mongo_client 
from credit_risk.entity.artifact_entity import DataIngestionArtifact

class DataIngestionArtifactData:
    def __init__(self):
        self.client = mongo_client
        self.database_name = "credit_artifact"
        self.collection_name = "Data-ingestion"
        self.collection = self.client[self.database_name][self.collection_name]

    def save_ingestion_artifact(self, data_ingestion_artifact: DataIngestionArtifact):
        self.collection.insert_one(data_ingestion_artifact.to_dict())

    def get_ingestion_artifact(self, query):
        self.collection.find_one(query)
