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

#model trainer
MODEL_TRAINER_BASE_ACCURACY = 0.7
MODEL_TRAINER_DIR = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_MODEL_NAME = "creditRisk_estimator"
MODEL_TRAINER_LABEL_INDEXER_DIR = "label_indexer"
MODEL_TRAINER_MODEL_METRIC_NAMES = ['f1',
                                    "weightedPrecision",
                                    "weightedRecall",
                                    "weightedTruePositiveRate",
                                    "weightedFalsePositiveRate",
                                    "weightedFMeasure",
                                    "truePositiveRateByLabel",
                                    "falsePositiveRateByLabel",
                                    "precisionByLabel",
                                    "recallByLabel",
                                    "fMeasureByLabel"]

#model evaluation
MODEL_EVALUATION_DIR = "model_evaluation"
MODEL_EVALUATION_REPORT_DIR = "report"
MODEL_EVALUATION_REPORT_FILE_NAME = "evaluation_report"
MODEL_EVALUATION_THRESHOLD_VALUE = 0.002
MODEL_EVALUATION_METRIC_NAMES = ['f1',]

#model push
MODEL_PUSHER_SAVED_MODEL_DIRS = "saved_models"
MODEL_PUSHER_DIR = "model_pusher"
MODEL_PUSHER_MODEL_NAME = MODEL_TRAINER_MODEL_NAME

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
 
class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig) -> None:
        try:
            #artifact/TIMESTAMP/model_trainer
            model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir,
                                                MODEL_TRAINER_DIR)
                                                
            #artifact/TIMESTAMP/model_trainer/creditRisk_estimator
            self.trained_model_file_path = os.path.join(model_trainer_dir, 
            MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_MODEL_NAME)
            
            #artifact/TIMESTAMP/model_trainer/label_indexer
            self.label_indexer_model_dir = os.path.join(
                model_trainer_dir, MODEL_TRAINER_LABEL_INDEXER_DIR
            )
            
            
            #base_accuracy = 0.7
            self.base_accuracy = MODEL_TRAINER_BASE_ACCURACY
            
            self.metric_list = MODEL_TRAINER_MODEL_METRIC_NAMES
        except Exception as e:
                raise CreditRiskException(e,sys)


class ModelEvaluationConfig:


    def __init__(self, training_pipeline_config:TrainingPipelineConfig) -> None:

        #artifact/TIMESTAMP/model_evaluation/
        self.model_evaluation_dir = os.path.join(training_pipeline_config.artifact_dir,
                                                MODEL_EVALUATION_DIR)
        #threshold = 0.02
        self.threshold=MODEL_EVALUATION_THRESHOLD_VALUE
        
        self.metric_list=MODEL_EVALUATION_METRIC_NAMES


class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):

        #artifact/TIMESTAMP/model_pusher/model/credit_riskEstimator
        self.pusher_model_dir = os.path.join(training_pipeline_config.artifact_dir,
                                                MODEL_PUSHER_DIR,"model",MODEL_PUSHER_MODEL_NAME)
        
        self.saved_model_dir = MODEL_PUSHER_SAVED_MODEL_DIRS

class BatchPredictionConfig:

    def __init__(self):
        try:
            self.inbox_dir = os.path.join("data","input-inbox")
            self.outbox_dir = os.path.join("data","output-outbox")
            self.archive_dir = os.path.join("data","archive")
            self.parquet_dir = os.path.join("data",'parquet_input')
            self.csv_dir = os.path.join("data","csv_output")
            os.makedirs(self.outbox_dir ,exist_ok=True)
            os.makedirs(self.archive_dir,exist_ok=True)
            os.makedirs(self.parquet_dir,exist_ok=True)
            os.makedirs(self.csv_dir,exist_ok=True)
        except Exception as e:
            raise CreditRiskException(e, sys)
