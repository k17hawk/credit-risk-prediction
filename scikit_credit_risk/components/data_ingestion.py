"""
author @ kumar dahal
this code is written to download the data from config/config.yaml
"""
from scikit_credit_risk.entity.config_entity import DataIngestionConfig
import sys,os
from scikit_credit_risk.exception import CreditException
from scikit_credit_risk import logging
from scikit_credit_risk.entity.artifact_entity import DataIngestionArtifact
import tarfile
import numpy as np
from six.moves import urllib
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import uuid
import json
import re
import time
class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CreditException(e,sys)
    
    
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            credit_file_path = os.path.join(raw_data_dir,file_name)
            df = pd.read_csv(credit_file_path)

            training_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

   
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)

            
            if training_data is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                training_data.to_csv(train_file_path,index=False)

            if test_data is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                test_data.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise CreditException(e,sys) from e
    

        
    def download_files(self, ):
        try:
            
             #extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url
            logging.info(f"Started downloading file from {download_url}")
            
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            # Ensure the raw_data_dir exists, if not, create it
            os.makedirs(raw_data_dir, exist_ok=True)

            # Define the file name and path to store the downloaded CSV
            file_name = os.path.basename(download_url)  # Extracts the file name from URL
            file_path = os.path.join(raw_data_dir, file_name)

            # Download the file
            response = requests.get(download_url)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Save the content to the file in raw_data_dir
            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Log completion of the download
            logging.info(f"File downloaded successfully and saved to {file_path}")

        except Exception as e:
            raise CreditException(e, sys)

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            tgz_file_path =  self.download_files()
            return self.split_data_as_train_test()
        except Exception as e:
            raise CreditException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")
