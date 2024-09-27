
from flask import Flask, request, render_template
from flask import jsonify
import pandas as pd
import os
from credit_risk.exception import CreditRiskException
from credit_risk.entity.config_entity import BatchPredictionConfig
from credit_risk.constant import TIMESTAMP
from datetime import datetime
from credit_risk.exception import CreditRiskException
from credit_risk.logger import logging 
from credit_risk.ml.esitmator import CreditRiskEstimator
from credit_risk.entity.schema import CreditRiskDataSchema
from credit_risk.config.spark_manager import spark_session
import os,sys
from credit_risk.entity.config_entity import BatchPredictionConfig
from pyspark.sql import DataFrame
import sys
import glob
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col,udf
from pyspark.sql.types import StringType

app = Flask(__name__)

class Application:
    def __init__(self) -> None:
        self.config = BatchPredictionConfig()
        self.setup_routes()  
        self.schema = CreditRiskDataSchema()

    def setup_routes(self):
        @app.route('/') 
        def form():
            return render_template('index.html') 
        @app.route('/submit', methods=['POST']) 
        def submit():
            try:
                data = {
                    "person_age": request.form.get('person_age'),
                    "person_income": request.form.get('person_income'),
                    "person_home_ownership": request.form.get('person_home_ownership'),
                    "person_emp_length": request.form.get('person_emp_length'),
                    "loan_intent": request.form.get('loan_intent'),
                    "loan_grade": request.form.get('loan_grade'),
                    "loan_amnt": request.form.get('loan_amnt'),
                    "loan_int_rate": request.form.get('loan_int_rate'),
                    "loan_percent_income": request.form.get('loan_percent_income'),
                    "cb_person_default_on_file": request.form.get('cb_person_default_on_file'),
                    "cb_person_cred_hist_length": request.form.get('cb_person_cred_hist_length')
                }
                
                df = pd.DataFrame([data]) 

                parquet_file_name = f"{TIMESTAMP}_single.parquet"
                parquet_file_path = os.path.join(self.config.parquet_dir, parquet_file_name)

                df.to_parquet(parquet_file_path, index=False)

                files = [f for f in os.listdir(self.config.parquet_dir) if f.endswith('.parquet')]
                files_with_paths = [os.path.join(self.config.parquet_dir, f) for f in files]

                latest_file = max(files_with_paths, key=os.path.getmtime)
                print(latest_file)
                encoded_value =  self.start_prediction(latest_file)
                if encoded_value==1.0:
                    value =  "Loan Granted"
                else:
                        value =  "Sorry Loan Not Granted.."

                return jsonify({"loan_status": value})

            except Exception as e:
                return jsonify({"error": str(e)})
            

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        try:
            csv_file_path = os.path.join(self.config.inbox_dir, filename)
            df.to_csv(csv_file_path, index=False)  #
            print(f"DataFrame saved to {csv_file_path}")
        except Exception as e:
            raise CreditRiskException(e, sys)
    

    def start_prediction(self,parquet_dir:str):
        credit_risk_estimator = CreditRiskEstimator()
        print(parquet_dir)

        df: DataFrame = spark_session.read.parquet(parquet_dir, multiline=True)
        df = df.withColumn("loan_int_rate", col("loan_int_rate").cast("bigint"))
        df = df.withColumn("person_emp_length", col("person_emp_length").cast("bigint"))
        
        print(f"Loaded DataFrame with {df.count()} rows.")
 
        logging.info("Saving preprocessed data.")

        print(f"Row: [{df.count()}] Column: [{len(df.columns)}]")
        print(f"Expected Column: {self.schema.required_columns_prediction}\nPresent Columns: {df.columns}")



        try:
            prediction_df = credit_risk_estimator.transform(dataframe=df)
            prediction_df.printSchema()
            print("data trasformed successfully")

            vector_columns = ['encoded_cb_person_default_on_file', 'encoded_person_home_ownership', 
                          'encoded_loan_intent', 'encodeed_loan_grade', 'encoded_income_group', 
                          'encoded_age_group', 'encoded_loan_amount_group', 'features', 
                          'scaled_output_features', 'rawPrediction', 'probability']
            
            for col_name in vector_columns:
                if col_name in prediction_df.columns:
                    prediction_df = prediction_df.drop(col_name)
            print("successfully dropped vector columns")
            
            csv_file_name = f'{TIMESTAMP}_predicted.csv'
            csv_file_path = os.path.join(self.config.csv_dir,csv_file_name)
            prediction_df.write.csv(csv_file_path, header=True, mode="overwrite")
            print("successfully stored")
            encoded_value = prediction_df.select('prediction_loan_status')
            return encoded_value

        except Exception as e:
            return jsonify({"error": str(e)})
        
    def run(self):
        self.app.run(debug=True) 


app_instance = Application()  

if __name__ == '__main__':
    app_instance.run(host='127.0.0.1', port=5003) 
