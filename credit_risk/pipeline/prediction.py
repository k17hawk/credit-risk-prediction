from credit_risk.exception import CreditRiskException
from credit_risk.logger import logging 
from credit_risk.ml.esitmator import CreditRiskEstimator
from credit_risk.config.spark_manager import spark_session
import os,sys
from credit_risk.entity.config_entity import BatchPredictionConfig
from credit_risk.constant import TIMESTAMP
from pyspark.sql import DataFrame
class Prediction:

    def __init__(self,batch_config:BatchPredictionConfig):
        try:
            self.batch_config=batch_config 
        except Exception as e:
            raise CreditRiskException(e, sys)
        
    def start_prediction(self):
        try:
            input_files = os.listdir(self.batch_config.inbox_dir)
            
            if len(input_files)==0:
                logging.info(f"No file found hence closing the batch prediction")
                return None 

            finance_estimator = CreditRiskEstimator()
            for file_name in input_files:
                data_file_path = os.path.join(self.batch_config.inbox_dir,file_name)
                df: DataFrame = spark_session.read.csv(data_file_path, header=True, inferSchema=True)
                
                

                prediction_df = finance_estimator.transform_with_stages(dataframe=df)
                prediction_file_path = os.path.join(self.batch_config.outbox_dir,f"{file_name}_{TIMESTAMP}")
                prediction_df.write.csv(prediction_file_path, header=True, mode='overwrite')


                archive_file_path = os.path.join(self.batch_config.archive_dir,f"{file_name}_{TIMESTAMP}")
                df.write.csv(archive_file_path)
        except Exception as e:
            raise CreditRiskException(e, sys)
