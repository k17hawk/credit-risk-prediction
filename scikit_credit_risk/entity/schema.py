from typing import List
from pyspark.sql.types import (TimestampType, 
            StringType, FloatType, StructType, StructField)
from pyspark.sql import DataFrame
import os, sys
from typing import Dict

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
        self.col_income_group = 'income_group'
        self.col_age_group = 'age_group'
        self.col_loan_amount_group = 'loan_amount_group'

        self.col_num__person_age = 'num__person_age'
        self.col_num__person_income = 'num__person_income'
        self.col_num__person_emp_length = 'num__person_emp_length'
        self.col_num__loan_amnt = 'num__loan_amnt'
        self.col_num__loan_int_rate = 'num__loan_int_rate'
        self.col_num__cb_person_cred_hist_length = 'num__cb_person_cred_hist_length'
        self.col_cat__cb_person_default_on_file_N = 'cat__cb_person_default_on_file_N'
        self.col_cat__cb_person_default_on_file_Y = 'cat__cb_person_default_on_file_Y'
        self.col_cat__person_home_ownership_MORTGAGE = 'cat__person_home_ownership_MORTGAGE'
        self.col_cat__person_home_ownership_OTHER = 'cat__person_home_ownership_OTHER'
        self.col_cat__person_home_ownership_OWN = 'cat__person_home_ownership_OWN'
        self.col_cat__person_home_ownership_RENT = 'cat__person_home_ownership_RENT'
        self.col_cat__loan_intent_DEBTCONSOLIDATION = 'cat__loan_intent_DEBTCONSOLIDATION'
        self.col_cat__loan_intent_EDUCATION = 'cat__loan_intent_EDUCATION'
        self.col_cat__loan_intent_HOMEIMPROVEMENT = 'cat__loan_intent_HOMEIMPROVEMENT'
        self.col_cat__loan_intent_MEDICAL  = 'cat__loan_intent_MEDICAL'
        self.col_cat__loan_intent_PERSONAL = 'cat__loan_intent_PERSONAL'
        self.col_cat__loan_intent_VENTURE = 'cat__loan_intent_VENTURE'
        self.col_cat__loan_grade_A = 'cat__loan_grade_A'
        self.col_cat__loan_grade_B = 'cat__loan_grade_B'
        self.col_cat__loan_grade_C = 'cat__loan_grade_C'
        self.col_cat__loan_grade_D = "cat__loan_grade_D"
        self.col_cat__loan_grade_E = 'cat__loan_grade_E'
        self.col_cat__loan_grade_F = 'cat__loan_grade_F'
        self.col_cat__loan_grade_G = 'cat__loan_grade_G'
        self.col_cat__income_group_high = 'cat__income_group_high'
        self.col_cat__income_group_high_middle = 'cat__income_group_high_middle'
        self.col_cat__income_group_low = 'cat__income_group_low'
        self.col_cat__income_group_low_middle = 'cat__income_group_low_middle'
        self.col_cat__income_group_middle = 'cat__income_group_middle'
        self.col_cat__age_group_20_25 = 'cat__age_group_20_25'
        self.col_cat__age_group_26_35 = 'cat__age_group_26_35'
        self.col_cat__age_group_36_45 = 'cat__age_group_36_45'
        self.col_cat__age_group_46_55 = 'cat__age_group_46_55'
        self.col_cat__age_group_56_65 = 'cat__age_group_56_65'
        self.col_cat__age_group_66_80 = 'cat__age_group_66_80'
        self.col_cat__loan_amount_group_high = 'cat__loan_amount_group_high'
        self.col_cat__loan_amount_group_medium = 'cat__loan_amount_group_medium'
        self.col_cat__loan_amount_group_small = 'cat__loan_amount_group_small'
        self.col_cat__loan_amount_group_very_high = 'cat__loan_amount_very_high'
        self.col_remainder__loan_percent_income = 'remainder__loan_percent_income'
        self.col_remainder__loan_to_income_ratio = 'remainder__loan_to_income_ratio'
        self.col_remainder__loan_to_emp_length_ratio = 'remainder__loan_to_emp_length_ratio'
        self.col_remainder__int_rate_to_loan_amt_ratio = 'remainder__int_rate_to_loan_amt_ratio'

        

    @property
    def required_scaling_columns(self) -> List[str]:
        features  =[self.col_person_age,
                    self.col_person_income,
                    self.col_person_emp_length,
                    self.col_loan_amnt,
                    self.col_loan_int_rate,
                    self.col_cb_person_cred_hist_length
        ]
        return features

   

        # cb_person_default_on_file columns
       


    @property
    def dataframe_schema(self) -> StructType:
        try:
            schema = StructType([
                StructField(self.col_person_age, StringType()),
                StructField(self.col_person_income, StringType()),
                StructField(self.col_person_emp_length, StringType()),
                StructField(self.col_loan_amnt, StringType()),
                StructField(self.col_loan_int_rate, StringType()),
                StructField(self.col_loan_status, StringType()),
                StructField(self.col_loan_percent_income, StringType()),
                StructField(self.col_cb_person_cred_hist_length, StringType()),
                StructField(self.col_person_home_ownership, StringType()),
                StructField(self.col_loan_intent, StringType()),
                StructField(self.col_loan_grade, StringType()),
                StructField(self.col_cb_person_default_on_file, StringType()),
                StructField(self.col_loan_to_income_ratio, StringType()),
                StructField(self.col_loan_to_emp_length_ratio, StringType()),
                StructField(self.col_int_rate_to_loan_amt_ratio, StringType()),
                StructField(self.col_income_group, StringType()),
                StructField(self.col_age_group, StringType()),
                StructField(self.col_loan_amount_group, StringType()),

            ])
            return schema
        except Exception as e:
            raise e
        
    @property
    def target_column(self) -> str:
        return self.col_loan_status
    
    @property
    def features_columns(self) -> List[str]:
        feature_columns = [

                self.col_loan_percent_income,
                self.col_loan_to_income_ratio,
        ]
        return feature_columns
        
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
    def one_hot_encoding_features_derived(self) -> List[str]:
        features = [
            self.col_income_group,
            self.col_age_group,
            self.col_loan_amount_group
        ]
        return features
    
    @property
    def one_hot_encoding_features(self) -> List[str]:
        features = [
            self.col_cb_person_default_on_file,
            self.col_person_home_ownership,
            self.col_loan_intent,
            self.col_loan_grade,

        ]
        return features
    
  


    @property
    def required_columns(self) -> List[str]:
        features = [self.target_column] + self.one_hot_encoding_features + self.numerical_columns
        return features
    
    @property
    def required_columns_prediction(self) -> List[str]:
        features =  self.one_hot_encoding_features + self.numerical_columns
        return features
    

    
    def required_oneHot_features(self) -> List[str]:
        features = self.one_hot_encoding_features + self.one_hot_encoding_features_derived
        return features
    

    

    
    
    @property
    def output_assambling_column(self) -> str:
        return 'features'
    
    @property
    def prediction_column_name(self) -> str:
        return "prediction"
    
    @property
    def target_indexed_label(self) -> str:
        return f"indexed_{self.target_column}"

    @property
    def prediction_label_column_name(self) -> str:
        return f"{self.prediction_column_name}_{self.target_column}"
    
    @property
    def min_max_features(self) -> str:
        return "scaled_output_features"
    
    @property
    def get_feature_list(self) -> List[str]:
        features = [
            self.col_num__person_age,
            self.col_num__person_income,
            self.col_num__person_emp_length,
            self.col_num__loan_amnt,
            self.col_num__loan_int_rate,
            self.col_num__cb_person_cred_hist_length,
            self.col_cat__cb_person_default_on_file_N,
            self.col_cat__cb_person_default_on_file_Y,
            self.col_cat__person_home_ownership_MORTGAGE,
            self.col_cat__person_home_ownership_OTHER,
            self.col_cat__person_home_ownership_OWN,
            self.col_cat__person_home_ownership_RENT,
            self.col_cat__loan_intent_DEBTCONSOLIDATION,
            self.col_cat__loan_intent_EDUCATION,
            self.col_cat__loan_intent_HOMEIMPROVEMENT,
            self.col_cat__loan_intent_MEDICAL,
            self.col_cat__loan_intent_PERSONAL,
            self.col_cat__loan_intent_VENTURE,
            self.col_cat__loan_grade_A,
            self.col_cat__loan_grade_B,
            self.col_cat__loan_grade_C,
            self.col_cat__loan_grade_D,
            self.col_cat__loan_grade_E,
            self.col_cat__loan_grade_F,
            self.col_cat__loan_grade_G,
            self.col_cat__income_group_high,
            self.col_cat__income_group_high_middle,
            self.col_cat__income_group_low,
            self.col_cat__income_group_low_middle,
            self.col_cat__income_group_middle,
            self.col_cat__age_group_20_25,
            self.col_cat__age_group_26_35,
            self.col_cat__age_group_36_45,
            self.col_cat__age_group_46_55,
            self.col_cat__age_group_56_65,
            self.col_cat__age_group_66_80,
            self.col_cat__loan_amount_group_high,
            self.col_cat__loan_amount_group_medium,
            self.col_cat__loan_amount_group_small,
            self.col_cat__loan_amount_group_very_high,
            self.col_remainder__loan_percent_income,
            self.col_remainder__loan_to_income_ratio,
            self.col_remainder__loan_to_emp_length_ratio,
            self.col_remainder__int_rate_to_loan_amt_ratio
        ]
        return features


    




        
    



