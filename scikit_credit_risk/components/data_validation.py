import os
import sys
from collections import namedtuple
from typing import List,Dict
import pandas as pd
from scikit_credit_risk.entity.config_entity import DataValidationConfig
from scikit_credit_risk.entity.schema import CreditRiskDataSchema
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import logging as logger
from scikit_credit_risk.entity.artifact_entity import DataIngestionArtifact
from scikit_credit_risk.entity.artifact_entity import DataValidationArtifact
from scikit_credit_risk.data_access.data_validation_artifact import DataValidationArtifactData
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
        

    def read_data(self)-> pd.DataFrame:
        try:
            dataframe: pd.DataFrame = pd.read_parquet(self.data_ingestion_artifact.feature_store_file_path)
            
            # Renaming the column
            dataframe = dataframe.rename(columns={'cb_person_cred_hist_length\r': 'cb_person_cred_hist_length'})
            
            logger.info(f"DataFrame is created using file: {self.data_ingestion_artifact.feature_store_file_path}")
            logger.info(f"Number of rows: {len(dataframe)} and columns: {len(dataframe.columns)}")
            
            return dataframe
        except Exception as e:
            raise CreditRiskException(e,sys) 
        
    
    @staticmethod
    def get_missing_report(dataframe: pd.DataFrame, )-> Dict[str,MissingReport]:
        try:
            missing_report: Dict[str, MissingReport] = {}
            logger.info(f"Preparing missing reports from each column")
            
            # In pandas, use len(dataframe) to get the number of rows
            number_of_row = len(dataframe)

            # Iterate over each column to calculate missing values
            for column in dataframe.columns:
                # Count missing values in the current column
                missing_row = dataframe[column].isnull().sum()
                
                # Calculate the percentage of missing values
                missing_percentage = (missing_row * 100) / number_of_row
                
                # Create a MissingReport object for the current column
                missing_report[column] = MissingReport(
                    total_row=number_of_row,
                    missing_row=missing_row,
                    missing_percentage=missing_percentage
                )
                
            logger.info(f"Missing report prepared: {missing_report}")
            return missing_report

        except Exception as e:
            raise CreditRiskException(e,sys) 

    
    def is_required_columns_exist(self, dataframe: pd.DataFrame):
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
            dataframe: pd.DataFrame = self.read_data()
            self.is_required_columns_exist(dataframe=dataframe)
            logger.info("Saving preprocessed data.")
            print(f"Row: [{dataframe.count()}] Column: [{len(dataframe.columns)}]")
            print(f"Expected Column: {self.schema.required_columns}\nPresent Columns: {dataframe.columns}")
            os.makedirs(self.data_validation_config.accepted_train_dir, exist_ok=True)
            accepted_file_path = os.path.join(self.data_validation_config.accepted_train_dir,
                                              self.data_validation_config.filename
                                              )
            dataframe.to_parquet(accepted_file_path,engine="pyarrow")
            artifact = DataValidationArtifact(accepted_file_path=accepted_file_path,
                                              rejected_dir=self.data_validation_config.rejected_traiin_dir
                                              )
            self.data_validation_artifact_data.save_validation_artifact(data_valid_artifact=artifact)
            logger.info(f"Data validation artifact: [{artifact}]")
            logger.info(f"{'>>' * 20} Data Validation completed.{'<<' * 20}")
            return artifact
        except Exception as e:
            raise CreditRiskException(e, sys)

