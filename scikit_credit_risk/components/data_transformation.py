from scikit_credit_risk.entity.schema import CreditRiskDataSchema
from scikit_credit_risk.exception import CreditRiskException
from scikit_credit_risk.logger import  logging as logger
from scikit_credit_risk.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from scikit_credit_risk.entity.config_entity import DataTransformationConfig

from scikit_credit_risk.ml.features import  AgeGroupCategorizer,IncomeGroupCategorizer,LoanAmountCategorizer,\
                                    RatioFeatureGenerator,DataCleaner,Upsampling

from scikit_credit_risk.constant import AGE_BINS,AGE_LABELS
from scikit_credit_risk.data_access.data_transformation_artifact import DataTransformationArtifactData
import os,sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib  


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

    
    def get_data_transformation_pipeline(self) -> Pipeline:
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

            # Define the preprocessor with a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', MinMaxScaler())
                    ]), self.schema.required_scaling_columns),
                    
                    ('cat', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]), self.schema.required_oneHot_features())
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
            
            # ratio_feature_generator = RatioFeatureGenerator()
            
            # stages = [
            #     data_cleaner,                
            #     age_group_categorizer,       
            #     income_group_categorizer,   
            #     loan_amount_categorizer,     
            #     ratio_feature_generator,           
            # ]

            
            # for im_one_hot_feature, string_indexer_col in zip(self.schema.required_oneHot_features(),
            #                                                   self.schema.string_indexer_one_hot_features):
            #     string_indexer = StringIndexer(inputCol=im_one_hot_feature, outputCol=string_indexer_col)
            #     stages.append(string_indexer)
            # onehot_encoded = On(inputCols=self.schema.string_indexer_one_hot_features,
            #                                        outputCols=self.schema.output_one_hot_encoded_feature)
            # stages.append(onehot_encoded)

            
        
            

            # min_max_scaler = MinMaxScaler(inputCol=self.schema.output_assambling_column,outputCol=self.schema.min_max_features)

            # assembler = VectorAssembler(inputCols=self.schema.assambling_columns, outputCol=self.schema.output_assambling_column)
            # stages.append(customTransformer)
            # stages.append(assembler)
            # stages.append(min_max_scaler)
        
            
             
            # pipeline = Pipeline(stages=stages)
            # logger.info(f"Data transformation pipeline: [{pipeline}]")
            # # print(pipeline.stages)
            return pipeline_steps
        except Exception as e:
            raise CreditRiskException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            dataframe: pd.DataFrame = self.read_data()
            logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")

            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ration: {1 - test_size}:{test_size}")

            X = dataframe.drop(columns=[self.schema.target_column])  
            y = dataframe[self.schema.target_column] 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            logger.info(f"Train dataset has number of row: [{X_train.shape[0]}] and"
                        f" column: [{len(X_train.columns)}]")

            logger.info(f"Test dataset has number of row: [{X_test.shape[0]}] and"
                        f" column: [{len(X_test.columns)}]")
            print("before pipeline")
    

            pipeline = self.get_data_transformation_pipeline()
            transformed_pipeline = pipeline.fit(X_train)
            transformed_df = transformed_pipeline.transform(X_train)
     
            column_transformer = transformed_pipeline.named_steps['preprocessor']

            feature_names = column_transformer.get_feature_names_out()

            transformed_df = pd.DataFrame(transformed_df,columns = feature_names)
            Y_train = y_train.reset_index(drop=True)

            # Concatenate transformed features and the target variable
            transformed_df = pd.concat([transformed_df, Y_train], axis=1)
            upsampler = Upsampling(target_column=self.schema.target_column)

            # Fit the upsampling model to the DataFrame
            upsampler.fit(transformed_df)

            transformed_trained_dataframe = upsampler.upsample()

            # Display the first few rows
 
            # transformed_trained_dataframe.printSchema()


            # datasampler = DataUpsampler(transformed_training_data=transformed_trained_dataframe,target_column=self.schema.target_column)

            # transformed_trained_dataframe = datasampler.upsample_data()

            transformed_test_dataframe = transformed_pipeline.transform(X_test)
            transformed_test_df = pd.DataFrame(transformed_test_dataframe,columns = feature_names)

            Y_test = y_test.reset_index(drop=True)

            # Concatenate transformed features and the target variable
            transformed_test = pd.concat([transformed_test_df, Y_test], axis=1)
   
            export_pipeline_file_path = os.path.join(self.data_tf_config.export_pipeline_dir)
            os.makedirs(export_pipeline_file_path,exist_ok=True)
            os.makedirs(self.data_tf_config.transformation_train_dir,exist_ok=True)
            os.makedirs(self.data_tf_config.transformation_test_dir,exist_ok=True)

            transformed_train_data_file_path = os.path.join(self.data_tf_config.transformation_train_dir,
                                                            self.data_tf_config.file_name) 
            
            transformed_test_data_file_path = os.path.join(self.data_tf_config.transformation_test_dir,
                                                            self.data_tf_config.file_name) 
            pipeline_file_path = os.path.join(export_pipeline_file_path, 'pipeline.pkl')
            logger.info(f"saving the pipeline at[{pipeline_file_path}]")

            joblib.dump(transformed_pipeline, pipeline_file_path) 
            

            logger.info(f"Saving transformed train data at: [{transformed_train_data_file_path}]")
            print(transformed_trained_dataframe.shape,len(transformed_trained_dataframe.columns))
            transformed_trained_dataframe.to_parquet(transformed_train_data_file_path,engine="pyarrow")

            logger.info(f"Saving transformed test data at: [{transformed_test_data_file_path}]")
            print(transformed_test.shape,len(transformed_test.columns))
 
            transformed_test.to_parquet(transformed_test_data_file_path,engine="pyarrow")
   


            data_tf_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_data_file_path,
                transformed_test_file_path=transformed_test_data_file_path,
                exported_pipeline_file_path=export_pipeline_file_path
            )
            self.data_transformation_data.save_transformation_artifact(data_transformation_artifact=data_tf_artifact)
            logger.info(f"{'>>' * 20} Data Transformation completed.{'<<' * 20}")
            return data_tf_artifact

         
        except  Exception as e:
            raise CreditRiskException(e, sys)
    



