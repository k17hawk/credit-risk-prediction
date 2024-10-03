import sys
import joblib
import shutil
import os
import time
from typing import List, Optional
import re
from credit_risk.logger import logging

from credit_risk.entity.schema import CreditRiskDataSchema
MODEL_SAVED_DIR="saved_models"
MODEL_NAME="credit_risk_estimator"
import pandas as pd

class ModelResolver:

    def __init__(self,model_dir=MODEL_SAVED_DIR,model_name=MODEL_NAME):
        try:
            self.model_dir = model_dir
            self.model_name = model_name
            os.makedirs(self.model_dir,exist_ok=True)
        except Exception as e:
            raise e
            
    @property
    def is_model_present(self)->bool:
        if self.get_best_model_path():
            return True
        return False

    def get_best_model_path(self,)->Optional[str]:
        try:
            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                return None
            timestamps = list(map(int,timestamps))
            latest_timestamp = max(timestamps)
            latest_model_path= os.path.join(self.model_dir,f"{latest_timestamp}",self.model_name)
            return latest_model_path
        except Exception as e:
            raise e


    @property
    def get_save_model_path(self)->bool:
        timestamp = str(time.time())
        last_index = timestamp.find(".")
        return os.path.join(self.model_dir,f"{timestamp[:last_index]}",self.model_name)



class CreditRiskEstimator:

    def __init__(self, model_dir=MODEL_SAVED_DIR,model_name=MODEL_NAME):
        try:
          
            self.model_resolver = ModelResolver(model_dir=model_dir,model_name=model_name)
            self.model_dir = model_dir
            self.loaded_model_path = None
            self.__loaded_model = None
        except Exception as e:
            raise e

    def get_model(self):
        try:
            latest_model_path = self.model_resolver.get_best_model_path()
            if latest_model_path != self.loaded_model_path:
                self.__loaded_model = joblib.load(latest_model_path)
                self.loaded_model_path = latest_model_path
            return self.__loaded_model
        except Exception as e:
            raise e


    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting transformation process...")
            model = self.get_model()
            logging.info("Model loaded successfully.")
            # print("loading the model stages.....")
            # print(model.stages)
            
            # Log the input DataFrame schema
            logging.info("Input DataFrame schema:")
            # for index, stage in enumerate(model.stages):
            #     print("applying",stage)
            #     dataframe.printSchema()
            #     dataframe = stage.transform(dataframe)
            #     dataframe.printSchema()
            #     print("successful",stage)  
            transformed_df = model.predict(dataframe)
        

            # Log the output DataFrame schema
            logging.info("Output DataFrame schema after transformation:")
            return transformed_df
        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            raise e
