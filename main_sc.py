
from flask import Flask, request, render_template
from flask import jsonify
import pandas as pd
import os
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.entity.config_entity import BatchPredictionConfig
from scikit_credit_risk.constant import TIMESTAMP
from datetime import datetime
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import logging 
from scikit_credit_risk.ml.esitmator import CreditRiskEstimator
from scikit_credit_risk.entity.schema import CreditRiskDataSchema
import pandas as pd
import os,sys


import sys
import glob

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

        df: pd.DataFrame = pd.read_parquet(parquet_dir, engine='pyarrow')
        # df = df.withColumn("loan_int_rate", col("loan_int_rate").cast("bigint"))
        # df = df.withColumn("person_emp_length", col("person_emp_length").cast("bigint"))
        
        print(f"Loaded DataFrame with {df.shape} rows.")
 
        logging.info("Saving preprocessed data.")

        print(f"Row: [{df.shape}] Column: [{len(df.columns)}]")
        print(f"Expected Column: {self.schema.required_columns_prediction}\nPresent Columns: {df.columns}")
        try:
            print('predicting...')
            prediction_df = credit_risk_estimator.transform(dataframe=df)
            print(prediction_df.head())
            print("data trasformed successfully")

            
            csv_file_name = f'{TIMESTAMP}_predicted.csv'
            csv_file_path = os.path.join(self.config.csv_dir,csv_file_name)
            prediction_df.to_csv(csv_file_path, header=True,)
            print("successfully stored")
            print(prediction_df.show())
            encoded_value = prediction_df.select('prediction_loan_status')
            print(encoded_value)
            return encoded_value

        except Exception as e:
            return jsonify({"error": str(e)})
        
    def run(self):
        app.run(debug=True) 


app_instance = Application()  

if __name__ == '__main__':
    app_instance.run(host='127.0.0.1', port=5003) 
