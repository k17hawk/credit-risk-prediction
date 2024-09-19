import os
import sys
from collections import namedtuple
from typing import List,Dict
from pyspark.sql import DataFrame
from pyspark.sql.functions import col,lit
from credit_risk.config.spark_manager import spark_session
from credit_risk.entity.config_entity import DataValidationConfig
from credit_risk.entity.schema import CreditRiskDataSchema
from credit_risk.exception import CreditRiskException
from credit_risk.logger import logging as logger
from credit_risk.entity.artifact_entity import DataIngestionArtifact
from credit_risk.entity.artifact_entity import DataValidationArtifact
from credit_risk.data_access.data_validation_artifact import DataValidationArtifactData
ERROR_MESSAGE  ='error_msg'
MissingReport = namedtuple("MissingReport",["total_row",'missing_row','missing_percentage'])

class DataValidation():
    def __init__(self,
                 data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 schema = CreditRiskDataSchema()
                 ):
        try:
            self.data_validation_artifact_data = DataValidationArtifactData()
            logger.info(f"{'>>' * 20}Starting data validation.{'<<' * 20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = schema

        except Exception as e:
            raise CreditRiskException(e,sys) from e
        

    def read_data(self)-> DataFrame:
        try:
            dataframe:DataFrame = spark_session.read.parquet(self.data_ingestion_artifact.feature_store_file_path)
            dataframe = dataframe.withColumnRenamed('cb_person_cred_hist_length\r', 'cb_person_cred_hist_length')
            logger.info(f"data frame is created using file{self.data_ingestion_artifact.feature_store_file_path}")
            logger.info(f"Number of row: {dataframe.count()} and column: {len(dataframe.columns)}")
            return dataframe
        except Exception as e:
            raise CreditRiskException(e,sys) 
        
    
    @staticmethod
    def get_missing_report(dataframe: DataFrame, )-> Dict[str,MissingReport]:
        try:
            missing_report:Dict[str:MissingReport] = dict()
            logger.info(f"Preparing missing reports from each column")
            number_of_row = dataframe.count()

            for column in dataframe.columns:
                missing_row = dataframe.filter(f"{column} is null").count()
                missing_percentage = (missing_row*100)/ number_of_row
                missing_report[column]  = MissingReport(total_row=number_of_row,
                                                       missing_row=missing_row,
                                                       missing_percentage=missing_percentage)
                logger.info(f"Missing report prepared: {missing_report}")
                return missing_report
        except Exception as e:
            raise CreditRiskException(e,sys) 

    
    def is_required_columns_exist(self, dataframe: DataFrame):
        try:
            columns = list(filter(lambda x: x in self.schema.required_columns,
                                  dataframe.columns))

            if len(columns) != len(self.schema.required_columns):
                raise Exception(f"Required column missing\n\
                 Expected columns: {self.schema.required_columns}\n\
                 Found columns: {columns}\
                 ")
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info(f"Initiating data preprocessing.")
            dataframe: DataFrame = self.read_data()
            self.is_required_columns_exist(dataframe=dataframe)
            logger.info("Saving preprocessed data.")
            print(f"Row: [{dataframe.count()}] Column: [{len(dataframe.columns)}]")
            print(f"Expected Column: {self.schema.required_columns}\nPresent Columns: {dataframe.columns}")
            os.makedirs(self.data_validation_config.accepted_train_dir, exist_ok=True)
            accepted_file_path = os.path.join(self.data_validation_config.accepted_train_dir,
                                              self.data_validation_config.filename
                                              )
            dataframe.write.parquet(accepted_file_path)
            artifact = DataValidationArtifact(accepted_file_path=accepted_file_path,
                                              rejected_dir=self.data_validation_config.rejected_traiin_dir
                                              )
            self.data_validation_artifact_data.save_validation_artifact(data_valid_artifact=artifact)
            logger.info(f"Data validation artifact: [{artifact}]")
            logger.info(f"{'>>' * 20} Data Validation completed.{'<<' * 20}")
            return artifact
        except Exception as e:
            raise CreditRiskException(e, sys)

