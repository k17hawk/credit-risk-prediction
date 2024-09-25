from credit_risk.entity.schema import CreditRiskDataSchema
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from credit_risk.config.spark_manager import spark_session
from credit_risk.exception import CreditRiskException
from credit_risk.logger import  logging as logger
from credit_risk.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from credit_risk.entity.config_entity import DataTransformationConfig
from pyspark.sql import DataFrame
from credit_risk.ml.features import  AgeGroupCategorizer,IncomeGroupCategorizer,LoanAmountCategorizer,\
                                    RatioFeatureGenerator,DataCleaner,DataUpsampler,CustomTransformer

from pyspark.sql.types import DoubleType
    
from pyspark.sql.functions import col, rand,count
import os,sys
from credit_risk.constant import AGE_BINS,AGE_LABELS
from credit_risk.data_access.data_transformation_artifact import DataTransformationArtifactData

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql import DataFrame

class OneHotEncoderPipeline:
    def __init__(self, input_cols: list, output_cols: list):
        if len(input_cols) != len(output_cols):
            raise ValueError("Input columns and output columns lists must have the same length.")
        
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.pipeline = None
        self.pipeline_model = None

    def fit(self, df: DataFrame) -> None:
        """Fit the pipeline on the provided DataFrame."""
        stages = []
        
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            indexer = StringIndexer(inputCol=input_col, outputCol=f"{input_col}_index")
            encoder = OneHotEncoder(inputCols=[f"{input_col}_index"], outputCols=[output_col])
            stages.extend([indexer, encoder])
        
        self.pipeline = Pipeline(stages=stages)
        self.pipeline_model = self.pipeline.fit(df)

    def save(self, path: str) -> None:
        """Save the fitted pipeline model to the specified path."""
        if self.pipeline_model is not None:
            self.pipeline_model.save(path)

    @classmethod
    def load(cls, path: str) -> 'OneHotEncoderPipeline':
        """Load a pipeline model from the specified path."""
        instance = cls(input_cols=[], output_cols=[])
        instance.pipeline_model = PipelineModel.load(path)
        return instance

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform the DataFrame using the fitted pipeline model."""
        if self.pipeline_model is None:
            raise Exception("Model must be fitted before calling transform.")
        return self.pipeline_model.transform(df)
    
# class OneHotEncoderCustom(Transformer, DefaultParamsWritable, DefaultParamsReadable):
#     def __init__(self, encoding_columns=None, predefined_columns=None):
#         super(OneHotEncoderCustom, self).__init__()
#         self.schema = CreditRiskDataSchema()
#         self.encoding_columns = Param(self, "encoding_columns", "column to be one-hot encoded")
#         self.predefined_columns = predefined_columns

#         if encoding_columns is not None:
#             self._set(encoding_columns=encoding_columns)

#     def _transform(self, df: DataFrame) -> DataFrame:
#         # Get the column name to encode
#         encoding_columns = self.getOrDefault(self.encoding_columns)

#         # Check if the encoding column exists in the DataFrame
#         if encoding_columns not in df.columns:
#             raise ValueError(f"Column {encoding_columns} does not exist in DataFrame. Available columns: {df.columns}")

#         # Get distinct values from the specified column (these are the categories for one-hot encoding)
#         distinct_values = df.select(encoding_columns).distinct().rdd.map(lambda x: x[0]).collect()

#         # Create one-hot encoded columns based on distinct values
#         for value in distinct_values:
#             if value is not None:
#                 # Create new columns like encoding_column_value with 1 if matches, else 0
#                 df = df.withColumn(f"{encoding_columns}_{value}", F.when(F.col(encoding_columns) == value, 1).otherwise(0))

#         # Ensure all predefined columns are present (e.g., in the prediction phase)
#         if self.predefined_columns:
#             for predefined_column in self.predefined_columns:
#                 if predefined_column not in df.columns:
#                     # Add the missing predefined column with a default value of 0
#                     df = df.withColumn(predefined_column, F.lit(0))

#         # Optionally drop the original column after one-hot encoding
#         df = df.drop(encoding_columns)

#         return df

# class OneHotEncoderCustom(Transformer, DefaultParamsReadable, DefaultParamsWritable):
#     def __init__(self, encoding_columns=None):
#         super(OneHotEncoderCustom, self).__init__()
    
#         self.encoding_columns = Param(self, "encoding_columns", "columns to be one-hot encoded")

#         if encoding_columns is not None:
#             self._set(encoding_columns=encoding_columns)

#     def _transform(self, df: DataFrame) -> DataFrame:
#         encoding_columns = self.getOrDefault(self.encoding_columns)

#         for column in encoding_columns:
#             distinct_values = df.select(column).distinct().rdd.map(lambda x: x[0]).collect()

#             # Create one-hot encoded columns only for existing distinct values
#             for value in distinct_values:
#                 if value is not None:  # Skip None values if present
#                     column_name = f"{column}_{value}"
#                     if column_name not in df.columns:  # Check if the column already exists
#                         df = df.withColumn(column_name, F.when(F.col(column) == value, 1).otherwise(0))
#                 else:
#                     # If value is None, we can simply initialize it to 0 for the new column
#                     column_name = f"{column}_None"
#                     if column_name not in df.columns:  # Check if the column already exists
#                         df = df.withColumn(column_name, F.lit(0))

#             # Optionally drop the original column if needed
#             df = df.drop(column)

#         return df



class MinMaxScalerCustom(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, columns_to_scale=None, min_val=-3, max_val=3):
        super(MinMaxScalerCustom, self).__init__()
        
        # Define parameters for columns to scale and the min/max values
        self.columns_to_scale = Param(self, "columns_to_scale", "Columns to be min-max scaled")
        self.min_val = Param(self, "min_val", "Minimum value of the scaled range")
        self.max_val = Param(self, "max_val", "Maximum value of the scaled range")
        
        # Set parameters if they are provided
        if columns_to_scale is not None:
            self._set(columns_to_scale=columns_to_scale)
        self._set(min_val=min_val)
        self._set(max_val=max_val)
    
    def _transform(self, df: DataFrame) -> DataFrame:
        """
        The transform method scales the columns provided using Min-Max scaling and drops the original columns.
        """
        columns_to_scale = self.getOrDefault(self.columns_to_scale)
        min_val = self.getOrDefault(self.min_val)
        max_val = self.getOrDefault(self.max_val)
        
        # List to store the original columns to be dropped
        original_columns = []
        
        for col_name in columns_to_scale:
            # Type cast columns to DoubleType if they are not already
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            
            # Get min and max for each column
            min_col, max_col = df.select(F.min(col_name), F.max(col_name)).first()
            
            # Ensure min and max are not the same to avoid division by zero
            if min_col == max_col:
                raise ValueError(f"Column {col_name} has the same min and max values, which cannot be scaled.")
            
            # Apply Min-Max scaling formula
            scaled_col = (F.col(col_name) - min_col) / (max_col - min_col) * (max_val - min_val) + min_val
            
            # Add the scaled column
            df = df.withColumn(f"scaled_{col_name}", scaled_col)
            
            # Store original column name for later removal
            original_columns.append(col_name)
        
        # Drop the original columns
        df = df.drop(*original_columns)
        
        return df

    # Helper method to get columns to scale
    def getColumnsToScale(self):
        return self.getOrDefault(self.columns_to_scale)

    # Helper method to get min/max values
    def getMinMaxValues(self):
        return self.getOrDefault(self.min_val), self.getOrDefault(self.max_val)

    
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
    
    def read_data(self) -> DataFrame:
        try:
            file_path = self.data_val_artifact.accepted_file_path
            dataframe: DataFrame = spark_session.read.parquet(file_path)
            # dataframe.printSchema()
            return dataframe
        except Exception as e:
            raise CreditRiskException(e, sys)

    
    def get_data_transformation_pipeline(self) -> Pipeline:
        try:

            onehot_encoded = OneHotEncoderPipeline(input_cols=self.schema.required_oneHot_features(),output_cols=)
           
            expected_columns_cb_file_default = [self.schema.col_cb_person_default_on_file_N,self.schema.col_cb_person_default_on_file_Y] 
            cb_file_hot_encoder = OneHotEncoderCustom(
                                encoding_columns=self.schema.col_cb_person_default_on_file, 
                                predefined_columns=expected_columns_cb_file_default
            )
            person_home_ownership_default = [self.schema.col_person_home_ownership_MORTGAGE,
                                             self.schema.col_person_home_ownership_RENT,
                                             self.schema.col_person_home_ownership_OTHER,
                                             self.schema.col_person_home_ownership_OWN]
            
            person_home_ownership_encoder = OneHotEncoderCustom(
                encoding_columns=self.schema.col_person_home_ownership,
                predefined_columns=person_home_ownership_default
            )
            
            
            loan_intent_default = [self.schema.col_loan_intent_DEBTCONSOLIDATION,
                                   self.schema.col_loan_intent_VENTURE,
                                   self.schema.col_loan_intent_PERSONAL,
                                   self.schema.col_loan_intent_EDUCATION,
                                   self.schema.col_loan_intent_HOMEIMPROVEMENT,
                                   self.schema.col_loan_intent_MEDICAL
                                   ]
            
            loan_intent_encoder = OneHotEncoderCustom(
                encoding_columns=self.schema.col_loan_intent,
                predefined_columns=loan_intent_default
            )
            

            loan_grade_default = [self.schema.col_loan_grade_F,
                                   self.schema.col_loan_grade_E,
                                   self.schema.col_loan_grade_B,
                                   self.schema.col_loan_grade_D,
                                   self.schema.col_loan_grade_C,
                                   self.schema.col_loan_grade_A,
                                   self.schema.col_loan_grade_G]
            loan_grade_encoder = OneHotEncoderCustom(
                encoding_columns=self.schema.col_loan_grade,
                predefined_columns=loan_grade_default
            )

            

            income_group_default = [self.schema.col_income_group_low_middle,
                                   self.schema.col_income_group_low,
                                   self.schema.col_income_group_high,
                                   self.schema.col_income_group_middle,
                                   self.schema.col_income_group_high_middle
                                   ]
            
            income_group_encoder = OneHotEncoderCustom(
                encoding_columns = self.schema.col_income_group,
                predefined_columns = income_group_default
            )

            


            age_group_default = [self.schema.col_age_group_26_35,
                                   self.schema.col_age_group_20_25,
                                   self.schema.col_age_group_46_55,
                                   self.schema.col_age_group_36_45,
                                   self.schema.col_age_group_66_80,
                                   self.schema.col_age_group_56_65
                                   ]
            
            age_group_encoder = OneHotEncoderCustom(
                encoding_columns=self.schema.col_age_group,
                predefined_columns=age_group_default
            )

            

            loan_amount_group_default = [self.schema.col_loan_amount_group_high,
                                   self.schema.col_loan_amount_group_medium,
                                   self.schema.col_loan_amount_group_very_high,
                                   self.schema.col_loan_amount_group_small
                                   ]
            
            loan_amount_group_encoder = OneHotEncoderCustom(
                encoding_columns=self.schema.col_loan_amount_group,
                predefined_columns=loan_amount_group_default
            )
            



            # Instantiate the custom Min-Max scaler with a specific range (-3 to 3)
            min_max_scaler = MinMaxScalerCustom(columns_to_scale=self.schema.required_scaling_columns, min_val=-3, max_val=3)


            data_cleaner = DataCleaner(
                                        age_column='person_age',
                                        age_threshold=80,
                                        column_to_drop='index',
                                        emp_length_column='person_emp_length',
                                        emp_length_threshold=60
                                    )
            age_group_categorizer = AgeGroupCategorizer(inputCol='person_age', outputCol='age_group', bins=AGE_BINS, labels=AGE_LABELS)
            income_group_categorizer = IncomeGroupCategorizer(inputCol='person_income', outputCol='income_group')
            loan_amount_categorizer = LoanAmountCategorizer(inputCol=self.schema.col_loan_amnt,outputCol=self.schema.col_loan_amount_group)
    

            ratio_feature_generator = RatioFeatureGenerator()
            custom_transformer = CustomTransformer()
            assembler = VectorAssembler(inputCols=self.schema.assambling_column, outputCol=self.schema.output_assambling_column)
        
            stages = [
                data_cleaner,                
                age_group_categorizer,       
                income_group_categorizer,   
                loan_amount_categorizer,     
                ratio_feature_generator,      
                cb_file_hot_encoder,
                person_home_ownership_encoder,
                loan_intent_encoder,
                loan_grade_encoder,
                income_group_encoder,
                age_group_encoder,
                loan_amount_group_encoder,            
                min_max_scaler,
                custom_transformer,
                assembler
            ]
             
            pipeline = Pipeline(stages=stages)
            logger.info(f"Data transformation pipeline: [{pipeline}]")
            # print(pipeline.stages)
            return pipeline
        except Exception as e:
            raise CreditRiskException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            dataframe: DataFrame = self.read_data()
            dataframe.printSchema()
            logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")

            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ration: {1 - test_size}:{test_size}")
            train_dataframe, test_dataframe = dataframe.randomSplit([1 - test_size, test_size])
            logger.info(f"Train dataset has number of row: [{train_dataframe.count()}] and"
                        f" column: [{len(train_dataframe.columns)}]")

            logger.info(f"Test dataset has number of row: [{test_dataframe.count()}] and"
                        f" column: [{len(test_dataframe.columns)}]")
            # loan_amount_categorizer = LoanAmountCategorizer(input_col=self.schema.col_loan_amnt,output_col=self.schema.col_loan_amount_group)
    

            # train_dataframe = loan_amount_categorizer.categorize(train_dataframe)
            # test_dataframe = loan_amount_categorizer.categorize(test_dataframe)


            pipeline = self.get_data_transformation_pipeline()
            transformed_pipeline = pipeline.fit(train_dataframe)
            transformed_trained_dataframe = transformed_pipeline.transform(train_dataframe)

            datasampler = DataUpsampler(transformed_training_data=transformed_trained_dataframe,target_column=self.schema.target_column)

            transformed_trained_dataframe = datasampler.upsample_data()

            transformed_test_dataframe = transformed_pipeline.transform(test_dataframe)

            export_pipeline_file_path = self.data_tf_config.export_pipeline_dir
            os.makedirs(export_pipeline_file_path,exist_ok=True)
            os.makedirs(self.data_tf_config.transformation_train_dir,exist_ok=True)
            os.makedirs(self.data_tf_config.transformation_test_dir,exist_ok=True)

            transformed_train_data_file_path = os.path.join(self.data_tf_config.transformation_train_dir,
                                                            self.data_tf_config.file_name) 
            
            transformed_test_data_file_path = os.path.join(self.data_tf_config.transformation_test_dir,
                                                            self.data_tf_config.file_name) 
            logger.info(f"saving the pipeline at[{export_pipeline_file_path}]")
            transformed_pipeline.save(export_pipeline_file_path)

            logger.info(f"Saving transformed train data at: [{transformed_train_data_file_path}]")
            print(transformed_trained_dataframe.count(),len(transformed_trained_dataframe.columns))
            transformed_trained_dataframe.write.parquet(transformed_train_data_file_path)

            logger.info(f"Saving transformed test data at: [{transformed_test_data_file_path}]")
            print(transformed_test_dataframe.count(),len(transformed_test_dataframe.columns))
            transformed_test_dataframe.write.parquet(transformed_test_data_file_path)

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
    



