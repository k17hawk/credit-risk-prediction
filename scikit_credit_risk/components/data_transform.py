from scikit_credit_risk.entity.schema import CreditRiskDataSchema
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import  logging as logger
from scikit_credit_risk.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from scikit_credit_risk.entity.config_entity import DataTransformationConfig

from scikit_credit_risk.ml.features import  AgeGroupCategorizer,IncomeGroupCategorizer,LoanAmountCategorizer,RatioFeatureGenerator,DataCleaner

from scikit_credit_risk.constant import AGE_BINS,AGE_LABELS
from scikit_credit_risk.data_access.data_transformation_artifact import DataTransformationArtifactData
from scikit_credit_risk.utils import save_numpy_array_data,save_object
import os,sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib  
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline 
from imblearn.over_sampling import SMOTE

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig,
                 schema = CreditRiskDataSchema()):
        try:
            logger.info(f"{'>>' * 20}Starting data transformation.{'<<' * 20}")
            self.data_transformation_data = DataTransformationArtifactData()
            self.data_val_artifact = data_validation_artifact
            self.data_tf_config = data_transformation_config
            self.schema = schema
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def read_data(self) -> pd.DataFrame:
        try:
            file_path = self.data_val_artifact.accepted_file_path
            dataframe: pd.DataFrame = pd.read_parquet(file_path)
            # dataframe.printSchema()
            return dataframe
        except Exception as e:
            raise CreditRiskException(e, sys)
    
    def get_data_transformation_pipeline(self) -> ColumnTransformer:
        try:
            data_cleaner = DataCleaner(
            age_column=self.schema.col_person_age,
            age_threshold=80,
            column_to_drop='index',
            emp_length_column=self.schema.col_person_emp_length,
            emp_length_threshold=60
            )

            age_group_categorizer = AgeGroupCategorizer(
                input_col=self.schema.col_person_age,
                output_col=self.schema.col_age_group,
                bins=AGE_BINS,
                labels=AGE_LABELS
            )

            income_group_categorizer = IncomeGroupCategorizer(
                input_col=self.schema.col_person_income,
                output_col=self.schema.col_income_group
            )

            loan_amount_categorizer = LoanAmountCategorizer(
                input_col=self.schema.col_loan_amnt,
                output_col=self.schema.col_loan_amount_group
            )
            ratio_feature_generator = RatioFeatureGenerator()
            
            data_cleaner_pipeline  =Pipeline([("data_cleaner",data_cleaner)])
            age_group_categorizer_pipeline  =Pipeline([("age_group_categorizer",age_group_categorizer)])
            income_group_categorizer_pipeline  =Pipeline([("income_group_categorizer",income_group_categorizer)])
            loan_amount_categorizer_pipeline  =Pipeline([("loan_amount_categorizer",loan_amount_categorizer)])
            ratio_feature_generator_pipeline  =Pipeline([("ratio_feature_generator",ratio_feature_generator)])
     
            
            num_pipeline = Pipeline([
                ('scaler', MinMaxScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler',MinMaxScaler())
                ]
            )

            # Build the ColumnTransformer with specific column mapping
            preprocessing = ColumnTransformer([
                ('data_cleaner_pipeline', data_cleaner_pipeline, [self.schema.col_person_age, self.schema.col_person_emp_length]),  # Specify columns for data cleaner
                ('age_group_categorizer_pipeline', age_group_categorizer_pipeline, [self.schema.col_person_age]),  # Specify column for age group categorization
                ('income_group_categorizer_pipeline', income_group_categorizer_pipeline, [self.schema.col_person_income]),  # Specify column for income group
                ('loan_amount_categorizer_pipeline', loan_amount_categorizer_pipeline, [self.schema.col_loan_amnt]),  # Specify column for loan amount
                ('ratio_feature_generator_pipeline', ratio_feature_generator_pipeline, [
                    self.schema.col_loan_amnt, 
                    self.schema.col_person_income, 
                    self.schema.col_person_emp_length, 
                    self.schema.col_loan_int_rate
                    ]
                ), 
                ('num_pipeline', num_pipeline, self.schema.required_scaling_columns),  # Specify numeric columns for scaling
                # ('cat_pipeline', cat_pipeline, self.schema.required_oneHot_features())  # Specify categorical columns for one-hot encoding
            ], remainder='passthrough')

            return preprocessing
        except Exception as e:
            raise CreditRiskException(e,sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            dataframe: pd.DataFrame = self.read_data()
            logger.info(f"Number of rows: [{dataframe.shape[0]}] and columns: [{dataframe.shape[1]}]")

            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ratio: {1 - test_size}:{test_size}")
            train_dataframe, test_dataframe = train_test_split(dataframe, test_size=test_size, random_state=42)

            input_feature_train_df = train_dataframe.drop(columns=self.schema.target_column,axis=1)
            target_feature_train_df = train_dataframe[self.schema.target_column]

            input_feature_test_df = test_dataframe.drop(columns=self.schema.target_column,axis=1)
            target_feature_test_df = test_dataframe[self.schema.target_column]
            

            preprocessing_obj = self.get_data_transformation_pipeline()
            

            logger.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            transformed_columns = []

            for name, pipeline, columns in preprocessing_obj.transformers_:
                if name != 'remainder':
                    # For pipelines with transformers
                    if hasattr(pipeline, 'get_feature_names_out'):
                        transformed_columns.extend(pipeline.get_feature_names_out(columns))
                    else:
                        transformed_columns.extend(columns)
                else:
                    # Handle remainder columns, which pass through unchanged
                    transformed_columns.extend(columns)

            # Create a DataFrame with transformed columns
            transformed_df = pd.DataFrame(input_feature_train_arr, columns=transformed_columns)

            # Inspect the resulting transformed DataFrame with new column names
            print(transformed_df.head())

            # smote = SMOTE(sampling_strategy='minority')
            # X_resampled, y_resampled = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
            

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(self.data_tf_config.transformation_train_dir,exist_ok=True)
            os.makedirs(self.data_tf_config.transformation_test_dir,exist_ok=True)
            
    

            transformed_train_data_file_path = os.path.join(self.data_tf_config.transformation_train_dir,
                                                            self.data_tf_config.file_name) 
            
            transformed_test_data_file_path = os.path.join(self.data_tf_config.transformation_test_dir,
                                                            self.data_tf_config.file_name) 
            
            train_file_name = os.path.basename(transformed_train_data_file_path).replace(".parquet",".npz")
            test_file_name = os.path.basename(transformed_test_data_file_path).replace(".parquet",".npz")


            transformed_train_file_path = os.path.join(self.data_tf_config.transformation_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(self.data_tf_config.transformation_test_dir, test_file_name)

            logger.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            export_pipeline_file_path = os.path.join(self.data_tf_config.export_pipeline_dir)
            os.makedirs(export_pipeline_file_path,exist_ok=True)
            logger.info(f"Saving preprocessing object.")
            save_object(file_path=export_pipeline_file_path,obj=preprocessing_obj)

            data_tf_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                exported_pipeline_file_path=export_pipeline_file_path
            )
            self.data_transformation_data.save_transformation_artifact(data_transformation_artifact=data_tf_artifact)
            logger.info(f"{'>>' * 20} Data Transformation completed.{'<<' * 20}")
            return data_tf_artifact

         
        except  Exception as e:
            raise CreditRiskException(e, sys)
    






        

