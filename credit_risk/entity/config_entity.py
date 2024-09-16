import os,sys
from dataclasses import dataclass

from credit_risk.exception import CreditRiskException
from credit_risk.constant import TIMESTAMP

#Dataingestion constants
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_DOWNLOADED_DATA_DIR = "downloaded_files"
DATA_INGESTION_FILE_NAME = "credit_risk_train"
DATA_INGESTION_FAILED_DIR = "failed_downloaded_files"

DATA_INGESTION_DATA_SOURCE_URL = f"https://raw.githubusercontent.com/k17hawk/credit-risk-prediction/main/Data/credit_risk_dataset.csv"

DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"

#DataValidation constants
DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_FILE_NAME = "credit_risk"
DATA_VALIDATION_ACCEPTED_DATA_DIR = "accepted_data"
DATA_VALIDATION_REJECTED_DATA_DIR = "rejected_data"


##data transformation constants
DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_PIPELINE_DIR = "transformed_pipeline"
DATA_TRANSFORMATION_TRAIN_DIR = "train"
DATA_TRANSFORMATION_FILE_NAME = "credit_risk"
DATA_TRANSFORMATION_TEST_DIR = "test"
DATA_TRANSFORMATION_TEST_SIZE = 0.2

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str = 'artifact'
    artifact_dir:str = os.path.join(pipeline_name,TIMESTAMP)


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            #artifact/TIMESTAMP/data_ingestion/
            data_ingestion_master_dir  = os.path.join(os.path.dirname(training_pipeline_config.artifact_dir),DATA_INGESTION_DIR)

             #artifact/TIMESTAMP/data_ingestion/TIMESTAMP
            self.data_ingestion_dir = os.path.join(data_ingestion_master_dir,TIMESTAMP)

             #artifact/TIMESTAMP/data_ingestion/TIMESTAMP/download_files
            self.download_dir=os.path.join(self.data_ingestion_dir, DATA_INGESTION_DOWNLOADED_DATA_DIR)

            #artifact/TIMESTAMP/data_ingestion/TIMESTAMP/failed_downloaded_files
            self.failed_dir =os.path.join(self.data_ingestion_dir, DATA_INGESTION_FAILED_DIR)

            #artifact/TIMESTAMP/data_ingestion/feature_store
            self.feature_store_dir=os.path.join(data_ingestion_master_dir, DATA_INGESTION_FEATURE_STORE_DIR)
            
            #car_price
            self.file_name = DATA_INGESTION_FILE_NAME
            

            #https://github.com/k17hawk/credit-risk-prediction/blob/main/Data/credit_risk_dataset.csv
            self.datasource_url = DATA_INGESTION_DATA_SOURCE_URL
            
        except Exception as e:
            raise CreditRiskException(e,sys)
        
class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        try:
            #artifact/TIMESTAMP/data_validation
            data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR)

            #artifact/TIMESTAMP/data_validation/accepted_dir
            self.accepted_train_dir = os.path.join(data_validation_dir,DATA_VALIDATION_ACCEPTED_DATA_DIR)
            
            #artifact/TIMESTAMP/data_validation/rejected_dir
            self.rejected_traiin_dir = os.path.join(data_validation_dir,DATA_VALIDATION_REJECTED_DATA_DIR)
            
            #car_price
            self.filename = DATA_VALIDATION_FILE_NAME

        except Exception as e:
            raise CreditRiskException(e,sys)
        
class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        try:
            #artifact/TIMESTAMP/data_transformation
            data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR)
            
            #artifact/TIMESTAMP/data_transformation/train
            self.transformation_train_dir = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRAIN_DIR)

            #artifact/TIMESTAMP/data_transformation/test
            self.transformation_test_dir = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TEST_DIR)

            #artifact/TIMESTAMP/data_transformation/transformed_pipeline
            self.export_pipeline_dir = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_PIPELINE_DIR)
            
            #file_name = credit_risk
            self.file_name = DATA_TRANSFORMATION_FILE_NAME

            #test_size = 0.2 
            self.test_size = DATA_TRANSFORMATION_TEST_SIZE


        except Exception as e:
                raise CreditRiskException(e,sys)
