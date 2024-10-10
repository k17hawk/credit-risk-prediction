"""
author: @ kumar dahal
this code is written to perform feature scaling and encoding
"""

from cgi import test
from sklearn import preprocessing
from scikit_credit_risk.exception import CreditException
from scikit_credit_risk import logging
from scikit_credit_risk.entity.config_entity import DataTransformationConfig 
from scikit_credit_risk.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scikit_credit_risk.constants import *
from scikit_credit_risk.utils.common import read_yaml_file,save_object,save_numpy_array_data,load_data
from sklearn.utils import resample
from sklearn.impute import SimpleImputer

class Upsampling:
    def __init__(self, target_column: str):
        """
        Initialize the upsampling class with the target column.

        Parameters:
        target_column (str): The name of the target variable in the DataFrame.
        """
        self.target_column = target_column

    def fit(self, dataframe: pd.DataFrame):
        """
        Fit the upsampling model to the provided DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.

        Returns:
        None
        """
        self.dataframe = dataframe
        self.majority_class = self.dataframe[self.target_column].value_counts().idxmax()
        self.minority_class = self.dataframe[self.target_column].value_counts().idxmin()

    def upsample(self):
        """
        Upsample the minority class in the DataFrame.

        Returns:
        pd.DataFrame: A new DataFrame with the upsampled minority class.
        """
        # Separate majority and minority classes
        majority = self.dataframe[self.dataframe[self.target_column] == self.majority_class]
        minority = self.dataframe[self.dataframe[self.target_column] == self.minority_class]

        # Upsample the minority class
        minority_upsampled = resample(minority,
                                       replace=True,  # Sample with replacement
                                       n_samples=len(majority),  # To match the majority class
                                       random_state=42)  # Reproducible results

        # Combine majority class with upsampled minority class
        upsampled_df = pd.concat([majority, minority_upsampled])

        # Shuffle the dataset
        upsampled_df = upsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return upsampled_df


class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, age_column:str=None, age_threshold:int=None, column_to_drop=None, emp_length_column:str=None, emp_length_threshold:int=None):
        self.age_column = age_column
        self.age_threshold = age_threshold
        self.column_to_drop = column_to_drop
        self.emp_length_column = emp_length_column
        self.emp_length_threshold = emp_length_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Check for nulls before dropping them
        if X.isnull().values.any():
            X = X.dropna()

        # Filter by age threshold if the column and threshold are set
        if self.age_column and self.age_threshold is not None:
            X.loc[:, self.age_column] = X[self.age_column].astype(int)
            max_age = X[self.age_column].max()
            if max_age > self.age_threshold:
                X = X[X[self.age_column] <= self.age_threshold]

        # Filter by employment length
        if self.emp_length_column and self.emp_length_threshold is not None:
            X = X[X[self.emp_length_column] <= self.emp_length_threshold]

        # Drop the specified column
        if self.column_to_drop and self.column_to_drop in X.columns:
            X = X.drop(columns=[self.column_to_drop])

        # Reset the index of the DataFrame
        X = X.reset_index(drop=True)

        return X


class AgeGroupCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col:str=None, output_col:str=None, bins=None, labels=None):
        self.input_col = input_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Ensure bins and labels are provided
        if self.bins is None or self.labels is None:
            raise ValueError("Both bins and labels must be provided.")
        
        # Check that the number of labels is one less than the number of bins
        if len(self.labels) != len(self.bins) - 1:
            raise ValueError("The number of labels must be equal to len(bins) - 1.")

        # Use pandas cut to assign age groups
        X[self.output_col] = pd.cut(X[self.input_col], bins=self.bins, labels=self.labels, right=False, include_lowest=True)
        X[self.output_col] = X[self.output_col].astype(str)

        return X
class IncomeGroupCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col:str=None, output_col:str=None):
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the input is a valid pandas DataFrame or Series
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Input must be a pandas DataFrame or Series.")
        
        # If X is a scalar value (a single income value), wrap it in a DataFrame
        if isinstance(X, pd.Series) or np.isscalar(X):
            X = pd.DataFrame({self.input_col: [X]})
        
        # Create a new column for the income group categorization
        X[self.output_col] = np.select(
            [
                X[self.input_col].between(0, 25000),
                X[self.input_col].between(25001, 50000),
                X[self.input_col].between(50001, 75000),
                X[self.input_col].between(75001, 100000)
            ],
            [
                'low',
                'low_middle',
                'middle',
                'high_middle'
            ],
            default='high'
        )
        
        return X
    
class LoanAmountCategorizer(BaseEstimator, TransformerMixin):
    def __init__(self, input_col:str = None, output_col:str=None):
        self.input_col = input_col
        self.output_col = output_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the input is a valid pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Create a new column for the loan amount group categorization
        X[self.output_col] = np.select(
            [
                X[self.input_col].between(0, 5000),
                X[self.input_col].between(5001, 10000),
                X[self.input_col].between(10001, 15000)
            ],
            [
                'small',
                'medium',
                'high'
            ],
            default='very_high'
        )
        
        return X
class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # No parameters needed for this transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the input is a valid pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Ensure required columns exist
        required_columns = ['loan_amnt', 'person_income', 'person_emp_length', 'loan_int_rate']
        for col in required_columns:
            if col not in X.columns:
                raise ValueError(f"Column '{col}' is not in the DataFrame.")

        # Create new ratio columns 
        X['loan_to_income_ratio'] = (X['loan_amnt'] / X['person_income'])
        X['loan_to_emp_length_ratio'] = (X['person_emp_length'] / X['loan_amnt'])
        X['int_rate_to_loan_amt_ratio'] = (X['loan_int_rate'] / X['loan_amnt'])

        return X
# class RatioFeatureGenerator(BaseEstimator, TransformerMixin):
#     def __init__(self, loan_to_income_col='loan_to_income_ratio', 
#                  loan_to_emp_length_col='loan_to_emp_length_ratio', 
#                  int_rate_to_loan_amt_col='int_rate_to_loan_amt_ratio'):
#         """
#         Initializes the RatioFeatureGenerator transformer with custom output column names.
        
#         Parameters:
#         loan_to_income_col: str, name for the loan to income ratio column.
#         loan_to_emp_length_col: str, name for the loan to employment length ratio column.
#         int_rate_to_loan_amt_col: str, name for the interest rate to loan amount ratio column.
#         """
#         self.loan_to_income_col = loan_to_income_col
#         self.loan_to_emp_length_col = loan_to_emp_length_col
#         self.int_rate_to_loan_amt_col = int_rate_to_loan_amt_col

#     def fit(self, X, y=None):
#         # Fit method doesn't change anything, it's needed for the pipeline compatibility
#         return self

#     def transform(self, X):
#         """
#         Transforms the input DataFrame by creating new ratio columns with specified names.

#         Parameters:
#         X : pandas DataFrame
#             The input DataFrame containing columns 'loan_amnt', 'person_income',
#             'person_emp_length', and 'loan_int_rate'.

#         Returns:
#         pandas DataFrame
#             The transformed DataFrame with new ratio columns.
#         """
#         # Check if the input is a valid pandas DataFrame
#         if not isinstance(X, pd.DataFrame):
#             raise ValueError("Input must be a pandas DataFrame.")

#         # Ensure required columns exist
#         required_columns = ['loan_amnt', 'person_income', 'person_emp_length', 'loan_int_rate']
#         missing_columns = [col for col in required_columns if col not in X.columns]
#         if missing_columns:
#             raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

#         # Create new ratio columns with safe division (handle division by zero)
#         X = X.copy()  
#         X[self.loan_to_income_col] = X['loan_amnt'] / X['person_income'].replace(0, 1)
#         X[self.loan_to_emp_length_col] = X['person_emp_length'] / X['loan_amnt'].replace(0, 1)
#         X[self.int_rate_to_loan_amt_col] = X['loan_int_rate'] / X['loan_amnt'].replace(0, 1)

#         return X


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise CreditException(e,sys) from e

    

    def get_data_transformer_object(self)->Pipeline:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)
            

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            print(numerical_columns)
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            print(categorical_columns)
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ('num', Pipeline(steps=[
            #             ('scaler', MinMaxScaler())
            #         ]), numerical_columns),
                    
            #         ('cat', Pipeline(steps=[
        
            #             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            #         ]), categorical_columns)
            #     ],
            #     remainder='passthrough'  
            # )



        
            # data_pipeline = Pipeline(steps=[
            #         ('data_cleaner', DataCleaner(
            #             age_column='person_age',
            #             age_threshold=80,
            #             column_to_drop='index',
            #             emp_length_column='person_emp_length',
            #             emp_length_threshold=60
            #         )),
            #         ("income_group_categorizer",IncomeGroupCategorizer(
            #             input_col='person_income',
            #             output_col='income_group'
            #         )),
            #         ('loan_amount_categorizer' , LoanAmountCategorizer(
            #             input_col='loan_amnt',
            #             output_col='loan_amount_group'
            #         )),
            #         ('ratio_feature_gen' , RatioFeatureGenerator(
            #             loan_to_income_col='loan_to_income_ratio', 
            #             loan_to_emp_length_col='loan_to_emp_length_ratio',
            #             int_rate_to_loan_amt_col='int_rate_to_loan_amt_ratio'
            #         )),
            #         ("preprocessor", preprocessor)

            #     ]
            #     )
            data_cleaner = DataCleaner(
            age_column='person_age',
            age_threshold=80,
            column_to_drop='index',
            emp_length_column='person_emp_length',
            emp_length_threshold=60
            )

            age_group_categorizer = AgeGroupCategorizer(
                input_col='person_age',
                output_col='age_group',
                bins=AGE_BINS,
                labels=AGE_LABELS
            )

            income_group_categorizer = IncomeGroupCategorizer(
                input_col="person_income",
                output_col='income_group'
            )

            loan_amount_categorizer = LoanAmountCategorizer(
                input_col='loan_amnt',
                output_col='loan_amount_group'
            )
            ratio_feature_generator = RatioFeatureGenerator()

            # Define the preprocessor with a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', MinMaxScaler())
                    ]), numerical_columns),
                    
                    ('cat', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
            
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]), categorical_columns)
                ],
                remainder='passthrough'  # Keep other columns unchanged
            )

            # Final pipeline combining all steps
            pipeline_steps = Pipeline(steps=[
                ("data_cleaner", data_cleaner),
                ("age_group_categorizer", age_group_categorizer),
                ("income_group_categorizer", income_group_categorizer),
                ("loan_amount_categorizer", loan_amount_categorizer),
                ("ratio_feature_generator",ratio_feature_generator),
                ("preprocessor", preprocessor)
            ])

            return pipeline_steps

        except Exception as e:
            raise CreditException(e,sys) from e   


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            pipeline_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            print("loading data")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            print("applying preprocessing into train and test")
 
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
     
            input_feature_train_arr=pipeline_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = pipeline_obj.transform(input_feature_test_df)
            print("applying fit transform")
            column_transformer = pipeline_obj.named_steps['preprocessor']

            feature_names = column_transformer.get_feature_names_out()
  
            transformed_df = pd.DataFrame(input_feature_train_arr,columns = feature_names)
            Y_train = target_feature_train_df.reset_index(drop=True)
            transformed_df = pd.concat([transformed_df, Y_train], axis=1)
            print("concatenated..")

            transformed_test_df = pd.DataFrame(input_feature_test_arr,columns = feature_names)
            y_test = target_feature_test_df.reset_index(drop=True)
            transformed_test_df = pd.concat([transformed_test_df, y_test], axis=1)
            upsampler = Upsampling(target_column=target_column_name)

            

            # Fit the upsampling model to the DataFrame
            upsampler.fit(transformed_df)

            transformed_trained_dataframe = upsampler.upsample()

            print("data has been upsampled")


            train_arr = np.array(transformed_trained_dataframe)

            test_arr = np.array(transformed_test_df)
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=pipeline_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")