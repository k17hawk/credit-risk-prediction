from typing import List
from pyspark.sql.types import (TimestampType, 
            StringType, FloatType, StructType, StructField)
from credit_risk.exception import CreditRiskException
from pyspark.sql import DataFrame
import os, sys
from typing import Dict
'''
['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_status', 'loan_percent_income', 'cb_person_cred_hist_length']
['person_home_ownership', 'loan_intent', 'loan_grade', 'loan_grade']
'''
class CreditRiskDataSchema:

    def __init__(self) -> None:
        self.col_person_age = "person_age"
        self.col_person_income = "person_income"
        self.col_person_emp_length = 'person_emp_length'
        self.col_loan_amnt = 'loan_amnt'
        self.col_loan_int_rate = 'loan_int_rate'
        self.col_loan_status = 'loan_status'
        self.col_loan_percent_income = 'loan_percent_income'
        self.col_cb_person_cred_hist_length = 'cb_person_cred_hist_length'
        self.col_person_home_ownership = 'person_home_ownership'
        self.col_loan_intent = 'loan_intent'
        self.col_loan_grade = 'loan_grade'
        self.col_cb_person_default_on_file = 'cb_person_default_on_file'
        self.col_loan_to_income_ratio = 'loan_to_income_ratio'
        self.col_loan_to_emp_length_ratio = 'loan_to_emp_length_ratio'
        self.col_int_rate_to_loan_amt_ratio = 'int_rate_to_loan_amt_ratio'
    


        
    @property
    def dataframe_schema(self)  -> StructType:
        try:
            schema = StructType([
                StructField(self.col_person_age,StringType()),
                StructField(self.col_person_income,StringType()),
                StructField(self.col_person_emp_length,StringType()),
                StructField(self.col_loan_amnt,StringType()),
                StructField(self.col_loan_int_rate,StringType()),
                StructField(self.col_loan_status,StringType()),
                StructField(self.col_loan_percent_income,StringType()),
                StructField(self.col_cb_person_cred_hist_length,StringType()),
                StructField(self.col_person_home_ownership,StringType()),
                StructField(self.col_loan_intent,StringType()),
                StructField(self.col_loan_grade,StringType()),
                StructField(self.col_cb_person_default_on_file,StringType()),
                StructField(self.col_loan_to_income_ratio,StringType()),
                StructField(self.col_loan_to_emp_length_ratio,StringType()),
                StructField(self.col_int_rate_to_loan_amt_ratio,StringType()),
                

            ])
            return schema
        except Exception as e:
            raise CreditRiskException(e,sys) from e
        
    @property
    def target_column(self) -> str:
        return self.col_loan_status
    
    @property
    def derieved_column(self) -> List[str]:
        features = [

        ]
        return features

    @property
    def categorical_features(self) -> List[str]:
        features = [
            self.col_person_home_ownership,
            self.col_loan_intent,
            self.col_loan_grade,
            self.col_cb_person_default_on_file,
            self.col_loan_to_income_ratio,
            self.col_loan_to_emp_length_ratio,
            self.col_int_rate_to_loan_amt_ratio

        ]
        return features
    
    @property
    def numerical_columns(self) -> List[str]:
        features = [
            self.col_person_age,
            self.col_person_income,
            self.col_person_emp_length,
            self.col_loan_amnt,
            self.col_loan_int_rate,
            self.col_loan_percent_income,
            self.col_cb_person_cred_hist_length
        ]
        return features

    @property
    def oneHot_encoding(self)-> List[str]:
        features = [
               f"enc_{col}" for col in self.categorical_features
        ]
        return features
    





        
    



