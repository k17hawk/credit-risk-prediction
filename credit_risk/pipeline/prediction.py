from credit_risk.exception import CreditRiskException
from credit_risk.logger import logging 
from credit_risk.ml.esitmator import CreditRiskEstimator
from credit_risk.entity.schema import CreditRiskDataSchema
from credit_risk.config.spark_manager import spark_session
import os,sys
from credit_risk.entity.config_entity import BatchPredictionConfig
from credit_risk.constant import TIMESTAMP
from pyspark.sql import DataFrame


class Prediction:

    def __init__(self, batch_config: BatchPredictionConfig,schema = CreditRiskDataSchema()):
        try:
            self.batch_config = batch_config 
            self.schema = schema
        except Exception as e:
            raise CreditRiskException(e, sys)

    def read_csv_and_convert_to_parquet(self):
        try:
            input_files = os.listdir(self.batch_config.inbox_dir)
            if len(input_files) == 0:
                print("No files found in the inbox directory.")
                return None
            
            for file_name in input_files:
                data_file_path = os.path.join(self.batch_config.inbox_dir, file_name)
                if file_name.endswith('.csv'):
                    # Reading CSV file
                    print(f"Reading CSV file: {data_file_path}")
                    df: DataFrame = spark_session.read.csv(data_file_path, header=True, inferSchema=True)
                    df = df.dropna()

                    # Convert to Parquet
                    parquet_file_path = os.path.join(self.batch_config.parquet_dir, f"{os.path.splitext(file_name)[0]}.parquet")
                    
                    # Check if the Parquet file already exists
                    if os.path.exists(parquet_file_path):
                        print(f"Parquet file already exists: {parquet_file_path}. Skipping this file.")
                        continue
                    
                    print(f"Saving as Parquet: {parquet_file_path}")
                    df.write.parquet(parquet_file_path)

        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def is_required_columns_exist(self, dataframe: DataFrame):
        try:
            columns = list(filter(lambda x: x in self.schema.required_columns_prediction,
                                  dataframe.columns))

            if len(columns) != len(self.schema.required_columns_prediction):
                raise Exception(f"Required column missing\n\
                 Expected columns: {self.schema.required_columns_prediction}\n\
                 Found columns: {columns}\
                 ")
        except Exception as e:
            raise CreditRiskException(e, sys)

    def start_prediction(self):
        try:
            input_files = os.listdir(self.batch_config.parquet_dir)
            
            if len(input_files) == 0:
                logging.info("No files found in the Parquet directory, closing the batch prediction.")
                return None 

            credit_risk_estimator = CreditRiskEstimator()
            for file_name in input_files:
                data_file_path = os.path.join(self.batch_config.parquet_dir, file_name)
                df: DataFrame = spark_session.read.parquet(data_file_path, multiline=True)
                df = df.withColumnRenamed('cb_person_cred_hist_length\r', 'cb_person_cred_hist_length')

                # Log the schema and data count
                df.printSchema()
                print(f"Loaded DataFrame with {df.count()} rows.")

                self.is_required_columns_exist(dataframe=df)
                logging.info("Saving preprocessed data.")
                print(f"Row: [{df.count()}] Column: [{len(df.columns)}]")
                print(f"Expected Column: {self.schema.required_columns_prediction}\nPresent Columns: {df.columns}")
                print(df.columns)

                # try:
                #     prediction_df = credit_risk_estimator.transform(dataframe=df)
                # except Exception as e:
                #     print(f"Error during transformation for {file_name}: {e}")
                #     continue  # Skip to the next file on error``

                # prediction_file_path = os.path.join(self.batch_config.outbox_dir, f"{os.path.splitext(file_name)[0]}_predicted_{TIMESTAMP}.parquet")

                # if os.path.exists(prediction_file_path):
                #     print(f"Prediction file already exists: {prediction_file_path}. Skipping this file.")
                #     continue
                
                # prediction_df.write.parquet(prediction_file_path)

                # # Archive the original data
                # archive_file_path = os.path.join(self.batch_config.archive_dir, f"{os.path.splitext(file_name)[0]}_archived_{TIMESTAMP}.parquet")
                # df.write.parquet(archive_file_path)
                
        except Exception as e:
            raise CreditRiskException(e, sys)
