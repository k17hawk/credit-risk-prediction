from credit_risk.exception import CreditRiskException
from credit_risk.logger import logging 
from credit_risk.ml.esitmator import CreditRiskEstimator
from credit_risk.entity.schema import CreditRiskDataSchema
from credit_risk.config.spark_manager import spark_session
import os,sys
from credit_risk.entity.config_entity import BatchPredictionConfig
from credit_risk.constant import TIMESTAMP
from pyspark.sql import DataFrame
from datetime import datetime
from credit_risk.config.spark_manager import spark_session
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
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
                    df.select("cb_person_default_on_file").show()
                    df = df.dropna()

                    parquet_file_name = f"{os.path.splitext(file_name)[0]}_{TIMESTAMP}.parquet"
                    parquet_file_path = os.path.join(self.batch_config.parquet_dir, parquet_file_name)

                    print(f"Saving as Parquet: {parquet_file_path}")
                    df.write.parquet(parquet_file_path)
                    
        except Exception as e:
            print(f"Error occurred: {e}")
            
    def is_required_columns_exist(self, dataframe: DataFrame):
        try:
            columns = list(filter(lambda x: x in self.schema.required_columns_prediction,
                                    dataframe.columns))
            if len(columns) != len(self.schema.required_columns_prediction):
                raise Exception(f"Required column missing \n Expected columns: {self.schema.required_columns_prediction} \n Found columns: {columns}")
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def drop_vector_columns(self,df: DataFrame) -> DataFrame:

        vector_columns = [col for col in df.columns if df.schema[col].dataType.typeName() == 'vector']
        double_columns = [col for col in df.columns if df.schema[col].dataType.typeName()=='double']
        
        # Drop the vector columns
        df = df.drop(*vector_columns)
        df = df.drop(*double_columns)
        
        return df

    def convert_parquet_to_csv(self,parquet_file_path: str, csv_file_path: str):
        try:
    
            df = spark_session.read.parquet(parquet_file_path)
  
            df = self.drop_vector_columns(df)

            # Write to CSV
            df.write.csv(csv_file_path, header=True)
            print(f"Converted {parquet_file_path} to {csv_file_path}")
        
        except Exception as e:
            print(f"Error converting Parquet to CSV: {e}")




    def start_prediction(self):
        try:
 
            files = [f for f in os.listdir(self.batch_config.parquet_dir) if f.endswith('.parquet')]
            if len(files) == 0:
                logging.info("No files found in the Parquet directory, closing the batch prediction.")
                return None 

            credit_risk_estimator = CreditRiskEstimator()

            files_with_paths = [os.path.join(self.batch_config.parquet_dir, f) for f in files]
            latest_file = max(files_with_paths, key=os.path.getmtime)

            df: DataFrame = spark_session.read.parquet(latest_file, multiline=True)
            print(f"Loaded DataFrame with {df.count()} rows.")

            self.is_required_columns_exist(dataframe=df)
            logging.info("Saving preprocessed data.")
            print(f"Row: [{df.count()}] Column: [{len(df.columns)}]")
            print(f"Expected Column: {self.schema.required_columns_prediction}\nPresent Columns: {df.columns}")
            # print(latest_file)

            try:
                prediction_df = credit_risk_estimator.transform(dataframe=df)
            except Exception as e:
                print(f"Error during transformation for {latest_file}: {e}")

            prediction_file_path = os.path.join(self.batch_config.outbox_dir, f"predicted_{TIMESTAMP}.parquet")
            csv_file_path = os.path.join(self.batch_config.csv_dir, f"predicted_{TIMESTAMP}.csv")

                
            prediction_df.write.parquet(prediction_file_path)
            df = spark_session.read.parquet(prediction_file_path)
            columns_to_convert = [
                "loan_int_rate", 
                "loan_percent_income", 
                "loan_to_income_ratio", 
                "loan_to_emp_length_ratio", 
                "int_rate_to_loan_amt_ratio", 
                "indexed_cb_person_default_on_file", 
                "indexed_person_home_ownership", 
                "indexed_loan_intent", 
                "indexed_loan_grade", 
                "indexed_income_group", 
                "indexed_age_group", 
                "indexed_loan_amount_group",
                "prediction"
            ]


            for col in columns_to_convert:
                df = df.withColumn(col, df[col].cast(FloatType()))

            columns_to_drop = [
                "encoded_loan_intent", 
                "encoded_person_home_ownership", 
                "encoded_cb_person_default_on_file",
                "probability",
                "rawPrediction",
                "scaled_output_features",
                "features",
                "encoded_age_group",
                "encoded_loan_amount_group",
                "encodeed_loan_grade",
                "encoded_income_group",
                "prediction_loan_status"

            ]
            df = df.drop(*columns_to_drop)
           
            new_data = df.toPandas()
            new_data.to_csv(csv_file_path,header=True)

        
            archive_file_path = os.path.join(self.batch_config.archive_dir, f"{os.path.splitext(latest_file)[0]}_archived_{TIMESTAMP}.parquet")
            df.write.parquet(archive_file_path)
                
            print("prediction completed..Kumar, You are god! your Bhanja is  Bad Boy, he is AmongUS player , he only play with Orangee..")
                
        except Exception as e:
            raise CreditRiskException(e, sys)
