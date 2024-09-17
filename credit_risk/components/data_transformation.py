from credit_risk.entity.schema import CreditRiskDataSchema
from pyspark.ml.feature import OneHotEncoder,VectorAssembler,StringIndexer,MinMaxScaler
from pyspark.ml.pipeline import Pipeline
from credit_risk.config.spark_manager import spark_session
from credit_risk.exception import CreditRiskException
from credit_risk.logger import  logging as logger
from credit_risk.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from credit_risk.entity.config_entity import DataTransformationConfig
from pyspark.sql import DataFrame
from credit_risk.ml.features import  AgeGroupCategorizer,IncomeGroupCategorizer,LoanAmountCategorizer,\
                                    RatioFeatureGenerator,DataCleaner,DataUpsampler

from pyspark.sql.types import DoubleType
    
from pyspark.sql.functions import col, rand,count
import os,sys
from credit_risk.constant import AGE_BINS,AGE_LABELS
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

import pyspark.sql.functions as F
from pyspark.ml.param.shared import Param

class OneHotEncoderCustom(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, encoding_columns=None):
        super(OneHotEncoderCustom, self).__init__()
    
        self.encoding_columns = Param(self, "encoding_columns", "columns to be one-hot encoded")

        if encoding_columns is not None:
            self._set(encoding_columns=encoding_columns)

    def _transform(self, df: DataFrame) -> DataFrame:
        encoding_columns = self.getOrDefault(self.encoding_columns)

        for column in encoding_columns:
            distinct_values = df.select(column).distinct().rdd.map(lambda x: x[0]).collect()

            # Create one-hot encoded columns
            for value in distinct_values:
                df = df.withColumn(f"{column}_{value}", F.when(F.col(column) == value, 1).otherwise(0))

            # Drop the original column
            df = df.drop(column)

        return df




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
           
            one_hot_encoder = OneHotEncoderCustom(encoding_columns=self.schema.required_oneHot_features)
            

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
            loan_amount_categorizer = LoanAmountCategorizer(inputCol='loan_amnt', outputCol='loan_amount_group')
            ratio_feature_generator = RatioFeatureGenerator()

        

            stages = [
                data_cleaner,                
                age_group_categorizer,       
                income_group_categorizer,   
                loan_amount_categorizer,     
                ratio_feature_generator,      
                one_hot_encoder,             
                min_max_scaler          
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
            logger.info(f"Number of row: [{dataframe.count()}] and column: [{len(dataframe.columns)}]")

            test_size = self.data_tf_config.test_size
            logger.info(f"Splitting dataset into train and test set using ration: {1 - test_size}:{test_size}")
            train_dataframe, test_dataframe = dataframe.randomSplit([1 - test_size, test_size])
            logger.info(f"Train dataset has number of row: [{train_dataframe.count()}] and"
                        f" column: [{len(train_dataframe.columns)}]")

            logger.info(f"Test dataset has number of row: [{test_dataframe.count()}] and"
                        f" column: [{len(test_dataframe.columns)}]")


            pipeline = self.get_data_transformation_pipeline()
            transformed_pipeline = pipeline.fit(train_dataframe)
            transformed_trained_dataframe = transformed_pipeline.transform(train_dataframe)
            #shuffling
            transformed_trained_dataframe = transformed_trained_dataframe.withColumn("random", F.rand(43))
            transformed_trained_dataframe = transformed_trained_dataframe.orderBy("random").drop("random")
            datasampler = DataUpsampler(transformed_training_data=transformed_trained_dataframe,target_column=self.schema.target_column)

            transformed_trained_dataframe = datasampler.upsample_data()



            transformed_test_dataframe = transformed_pipeline.transform(test_dataframe)
            # duplicates = [col for col in transformed_trained_dataframe.columns if transformed_trained_dataframe.columns.count(col) > 1]
            #replacing the columns
            transformed_trained_dataframe = transformed_trained_dataframe.select([F.col(c).alias(c.replace(" ", "_")) for c in transformed_trained_dataframe.columns])
            
            transformed_test_dataframe = transformed_test_dataframe.select([F.col(c).alias(c.replace(" ", "_")) for c in transformed_test_dataframe.columns])

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
            logger.info(f"{'>>' * 20} Data Transformation completed.{'<<' * 20}")
            return data_tf_artifact

         
        except  Exception as e:
            raise CreditRiskException(e, sys)
    



