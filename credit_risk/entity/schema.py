from typing import List
from pyspark.sql.types import (TimestampType, 
            StringType, FloatType, StructType, StructField)
from credit_risk.exception import CreditRiskException
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

        # cb_person_default_on_file columns
        self.col_cb_person_default_on_file_Y = 'cb_person_default_on_file_Y'
        self.col_cb_person_default_on_file_N = 'cb_person_default_on_file_N'
        
        # person_home_ownership columns
        self.col_person_home_ownership_OWN = 'person_home_ownership_OWN'
        self.col_person_home_ownership_RENT = 'person_home_ownership_RENT'
        self.col_person_home_ownership_MORTGAGE = 'person_home_ownership_MORTGAGE'
        self.col_person_home_ownership_OTHER = 'person_home_ownership_OTHER'

        # loan_intent columns
        self.col_loan_intent_DEBTCONSOLIDATION = 'loan_intent_DEBTCONSOLIDATION'
        self.col_loan_intent_VENTURE = 'loan_intent_VENTURE'
        self.col_loan_intent_PERSONAL = 'loan_intent_PERSONAL'
        self.col_loan_intent_EDUCATION = 'loan_intent_EDUCATION'
        self.col_loan_intent_HOMEIMPROVEMENT = 'loan_intent_HOMEIMPROVEMENT'
        self.col_loan_intent_MEDICAL = 'loan_intent_MEDICAL'

        # loan_grade columns
        self.col_loan_grade_F = 'loan_grade_F'
        self.col_loan_grade_E = 'loan_grade_E'
        self.col_loan_grade_B = 'loan_grade_B'
        self.col_loan_grade_D = 'loan_grade_D'
        self.col_loan_grade_C = 'loan_grade_C'
        self.col_loan_grade_A = 'loan_grade_A'
        self.col_loan_grade_G = 'loan_grade_G'

        # income_group columns
        self.col_income_group_low_middle = 'income_group_low-middle'
        self.col_income_group_low = 'income_group_low'
        self.col_income_group_high = 'income_group_high'
        self.col_income_group_middle = 'income_group_middle'
        self.col_income_group_high_middle = 'income_group_high-middle'

        # age_group columns
        self.col_age_group_26_35 = 'age_group_26-35'
        self.col_age_group_20_25 = 'age_group_20-25'
        self.col_age_group_46_55 = 'age_group_46-55'
        self.col_age_group_36_45 = 'age_group_36-45'
        self.col_age_group_66_80 = 'age_group_66-80'
        self.col_age_group_56_65 = 'age_group_56-65'

        # loan_amount_group columns
        self.col_loan_amount_group_high = 'loan_amount_group_high'
        self.col_loan_amount_group_medium = 'loan_amount_group_medium'
        self.col_loan_amount_group_very_high = 'loan_amount_group_very_high'
        self.col_loan_amount_group_small = 'loan_amount_group_small'

        # scaled columns
        self.col_scaled_person_age = 'scaled_person_age'
        self.col_scaled_person_income = 'scaled_person_income'
        self.col_scaled_person_emp_length = 'scaled_person_emp_length'
        self.col_scaled_loan_amnt = 'scaled_loan_amnt'
        self.col_scaled_loan_int_rate = 'scaled_loan_int_rate'
        self.col_scaled_cb_person_cred_hist_length = 'scaled_cb_person_cred_hist_length'
        self.col_scaled_loan_to_emp_length_ratio = 'scaled_loan_to_emp_length_ratio'
        self.col_scaled_int_rate_to_loan_amt_ratio = 'scaled_int_rate_to_loan_amt_ratio'


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

                # cb_person_default_on_file columns
                StructField(self.col_cb_person_default_on_file_Y, StringType()),
                StructField(self.col_cb_person_default_on_file_N, StringType()),

                # person_home_ownership columns
                StructField(self.col_person_home_ownership_OWN, StringType()),
                StructField(self.col_person_home_ownership_RENT, StringType()),
                StructField(self.col_person_home_ownership_MORTGAGE, StringType()),
                StructField(self.col_person_home_ownership_OTHER, StringType()),

                # loan_intent columns
                StructField(self.col_loan_intent_DEBTCONSOLIDATION, StringType()),
                StructField(self.col_loan_intent_VENTURE, StringType()),
                StructField(self.col_loan_intent_PERSONAL, StringType()),
                StructField(self.col_loan_intent_EDUCATION, StringType()),
                StructField(self.col_loan_intent_HOMEIMPROVEMENT, StringType()),
                StructField(self.col_loan_intent_MEDICAL, StringType()),

                # loan_grade columns
                StructField(self.col_loan_grade_F, StringType()),
                StructField(self.col_loan_grade_E, StringType()),
                StructField(self.col_loan_grade_B, StringType()),
                StructField(self.col_loan_grade_D, StringType()),
                StructField(self.col_loan_grade_C, StringType()),
                StructField(self.col_loan_grade_A, StringType()),
                StructField(self.col_loan_grade_G, StringType()),

                # income_group columns
                StructField(self.col_income_group_low_middle, StringType()),
                StructField(self.col_income_group_low, StringType()),
                StructField(self.col_income_group_high, StringType()),
                StructField(self.col_income_group_middle, StringType()),
                StructField(self.col_income_group_high_middle, StringType()),

                # age_group columns
                StructField(self.col_age_group_26_35, StringType()),
                StructField(self.col_age_group_20_25, StringType()),
                StructField(self.col_age_group_46_55, StringType()),
                StructField(self.col_age_group_36_45, StringType()),
                StructField(self.col_age_group_66_80, StringType()),
                StructField(self.col_age_group_56_65, StringType()),

                # loan_amount_group columns
                StructField(self.col_loan_amount_group_high, StringType()),
                StructField(self.col_loan_amount_group_medium, StringType()),
                StructField(self.col_loan_amount_group_very_high, StringType()),
                StructField(self.col_loan_amount_group_small, StringType()),

                # scaled columns
                StructField(self.col_scaled_person_age, StringType()),
                StructField(self.col_scaled_person_income, StringType()),
                StructField(self.col_scaled_person_emp_length, StringType()),
                StructField(self.col_scaled_loan_amnt, StringType()),
                StructField(self.col_scaled_loan_int_rate, StringType()),
                StructField(self.col_scaled_cb_person_cred_hist_length, StringType()),
                StructField(self.col_scaled_loan_to_emp_length_ratio, StringType()),
                StructField(self.col_scaled_int_rate_to_loan_amt_ratio, StringType())
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
        self.col_cb_person_default_on_file_Y,
        self.col_cb_person_default_on_file_N ,
        
        # person_home_ownership columns
        self.col_person_home_ownership_OWN,
        self.col_person_home_ownership_RENT,
        self.col_person_home_ownership_MORTGAGE ,
        self.col_person_home_ownership_OTHER,


        # loan_intent columns
        self.col_loan_intent_DEBTCONSOLIDATION,
        self.col_loan_intent_VENTURE,
        self.col_loan_intent_PERSONAL,
        self.col_loan_intent_EDUCATION,
        self.col_loan_intent_HOMEIMPROVEMENT,
        self.col_loan_intent_MEDICAL,
        # loan_grade columns
        self.col_loan_grade_F,
        self.col_loan_grade_E,
        self.col_loan_grade_B ,
        self.col_loan_grade_D,
        self.col_loan_grade_C,
        self.col_loan_grade_A,
        self.col_loan_grade_G ,
        # income_group columns
        self.col_income_group_low_middle ,
        self.col_income_group_low ,
        self.col_income_group_high ,
        self.col_income_group_middle ,
        self.col_income_group_high_middle,

        # age_group columns
        self.col_age_group_26_35,
        self.col_age_group_20_25,
        self.col_age_group_46_55,
        self.col_age_group_36_45,
        self.col_age_group_66_80,
        self.col_age_group_56_65 ,

        # loan_amount_group columns
        self.col_loan_amount_group_high,
        self.col_loan_amount_group_medium ,
        self.col_loan_amount_group_very_high,
        self.col_loan_amount_group_small,

        # scaled columns
        self.col_scaled_person_age ,
        self.col_scaled_person_income,
        self.col_scaled_person_emp_length ,
        self.col_scaled_loan_amnt ,
        self.col_scaled_loan_int_rate ,
        self.col_scaled_cb_person_cred_hist_length ,
        self.col_scaled_loan_to_emp_length_ratio ,
        self.col_scaled_int_rate_to_loan_amt_ratio,
        ]
        return features
    
                # 'loan_status', 'loan_percent_income', 'loan_to_income_ratio', 'cb_person_default_on_file_Y',
# 'cb_person_default_on_file_N', 'person_home_ownership_OWN', 'person_home_ownership_RENT', 'person_home_ownership_MORTGAGE', 
# 'person_home_ownership_OTHER', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_VENTURE', 'loan_intent_PERSONAL', 'loan_intent_EDUCATION',\
#       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_grade_F', 'loan_grade_E', 'loan_grade_B', 'loan_grade_D', 'loan_grade_C',
# 'loan_grade_A', 'loan_grade_G', 'income_group_low-middle', 'income_group_low', 'income_group_high', 'income_group_middle', 'income_group_high-middle', 
# 'age_group_26-35', 'age_group_20-25', 'age_group_46-55', 'age_group_36-45', 'age_group_66-80', 'age_group_56-65', 'loan_amount_group_high',
# 'loan_amount_group_medium', 'loan_amount_group_very_high', 'loan_amount_group_small', 'scaled_person_age', 'scaled_person_income', 'scaled_person_emp_length',
# 'scaled_loan_amnt', 'scaled_loan_int_rate', 'scaled_cb_person_cred_hist_length', 'scaled_loan_to_emp_length_ratio', 'scaled_int_rate_to_loan_amt_ratio'
  
    @property
    def features_columns(self) -> List[str]:
        feature_columns = [

                self.col_loan_percent_income,
                self.col_loan_to_income_ratio,
        ]
        return feature_columns
        


    @property
    def assambling_column(self) -> List[str]:
        return self.features_columns+self.derieved_column







































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
    def output_assambling_column(self):
        return 'features'

    @property
    def required_columns(self) -> List[str]:
        features = [self.target_column] + self.one_hot_encoding_features + self.numerical_columns
        return features
    
    @property
    def required_columns_prediction(self) -> List[str]:
        features =  self.one_hot_encoding_features + self.numerical_columns
        return features
    

    @property
    def required_oneHot_features(self) -> List[str]:
        features  =self.one_hot_encoding_features + self.one_hot_encoding_features_derived
        return features
    
    @property
    def required_scaling_columns(self) -> List[str]:
        features  =[self.col_person_age,
                    self.col_person_income,
                    self.col_person_emp_length,
                    self.col_loan_amnt,
                    self.col_loan_int_rate,
                    self.col_cb_person_cred_hist_length,
                    self.col_loan_to_emp_length_ratio,
                    self.col_int_rate_to_loan_amt_ratio]
        return features


    @property
    def prediction_column_name(self) -> str:
        return "prediction"

    @property
    def prediction_label_column_name(self) -> str:
        return f"{self.prediction_column_name}_{self.target_column}"
    
    @property
    def scaled_vector_input_features(self) -> str:
        return "scaled_input_features"
    





        
    



