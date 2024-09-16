from collections import namedtuple
from credit_risk.entity import DataIngestionConfig
from credit_risk.logger import logging as logger
from credit_risk.exception import CreditRiskException
from typing import List
import sys,os
import requests
import uuid
import json
import re
import time
from credit_risk.entity import DataIngestionArtifact

Download_url = namedtuple('downloadURL',['url',
                                         'file_path',
                                         'n_retry'])
class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig,n_retry: int = 5):
        try:
            logger.info(f"{'>>' * 20}Starting data ingestion.{'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
            self.failed_download_urls: List[Download_url] = []
            self.n_retry = n_retry
        except Exception as e:
            raise CreditRiskException(e,sys)
        
    def retry_download_data(self,data,download_url:Download_url):
        try:
            # if retry still possible try else return the response
            if download_url.n_retry == 0:
                self.failed_download_urls.append(download_url)
                logger.info(f"Unable to download file {download_url.url}")
                return
            
            content = data.content.decode("utf-8")
            wait_second = re.findall(r'\d+', content)

            if len(wait_second) > 0:
                time.sleep(int(wait_second[0]) + 2)

            
            #artifact/TIMESTAMP/data_ingestion/TIMESTAMP/failed_downloaded_files/base_name
            failed_file_path = os.path.join(self.data_ingestion_config.failed_dir,
                                            os.path.basename(download_url.file_path))
            os.makedirs(self.data_ingestion_config.failed_dir, exist_ok=True)

            with open(failed_file_path, "wb") as file_obj:
                file_obj.write(data.content)

            download_url = Download_url(download_url.url, file_path=download_url.file_path,
                                       n_retry=download_url.n_retry - 1)
            self.download_files(download_url=download_url)

        except Exception as e:
            raise CreditRiskException(e,sys)
        
    def download_files(self, download_url: Download_url):
        try:
            logger.info(f"Started downloading file from {download_url.url}")
            
            download_dir = os.path.dirname(download_url.file_path)
            
            os.makedirs(download_dir, exist_ok=True)

            data = requests.get(download_url.url, params={'User-agent': f'your bot {uuid.uuid4()}'})

            try:
                logger.info(f"Started writing downloaded data into CSV file: {download_url.file_path}")
                
                with open(download_url.file_path, "wb") as file_obj: 
                    file_obj.write(data.content)
                    
                logger.info(f"Downloaded CSV data has been written into file: {download_url.file_path}")
                    
            except Exception as e:
                logger.info("Failed to download, retrying again.")
                if os.path.exists(download_url.file_path):
                    os.remove(download_url.file_path)
                self.retry_download_data(data, download_url=download_url)

        except Exception as e:
            raise CreditRiskException(e, sys)


        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(f"Starting the file download")
            
            download_url = Download_url(
                url=self.data_ingestion_config.datasource_url,
                file_path=os.path.join(self.data_ingestion_config.download_dir, self.data_ingestion_config.file_name + ".csv"),
                n_retry=self.n_retry
            )

            self.download_files(download_url)
            feature_store_file_path = os.path.join(self.data_ingestion_config.feature_store_dir,
                                                   self.data_ingestion_config.file_name)

            
            artifact =  DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
                download_dir=self.data_ingestion_config.download_dir
            )
            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact
        
        except Exception as e:
            raise CreditRiskException(e, sys)