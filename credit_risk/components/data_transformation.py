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
from pyspark.ml.feature import MinMaxScaler
    
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


# class CustomOneHotEncoder(Transformer, DefaultParamsWritable, DefaultParamsReadable):
#     def __init__(self, input_cols=None, output_cols=None):
#         super(CustomOneHotEncoder, self).__init__()
#         self.input_cols = Param(self, "input_cols", "")
#         self.output_cols = Param(self, "output_cols", "")

#         self._setDefault(input_cols=[], output_cols=[])

#         if input_cols is not None:
#             self._set(input_cols=input_cols)
#         if output_cols is not None:
#             self._set(output_cols=output_cols)

#     def _transform(self, df: DataFrame) -> DataFrame:
#         input_cols = self.getOrDefault(self.input_cols)
#         output_cols = self.getOrDefault(self.output_cols)

#         # Apply StringIndexer
#         indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in input_cols]
#         for indexer in indexers:
#             df = indexer.fit(df).transform(df)

#         # Apply OneHotEncoder
#         encoders = [OneHotEncoder(inputCols=[f"{col}_index"], outputCols=[output_col]) for col, output_col in zip(input_cols, output_cols)]
#         for encoder in encoders:
#             df = encoder.fit(df).transform(df)

#         for col in input_cols:
#             df = df.drop(col)
#         for col in [f"{col}_index" for col in input_cols]:
#             df = df.drop(col)


#         return df


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
            customTransformer = CustomTransformer()
    

            ratio_feature_generator = RatioFeatureGenerator()
            stages = [
                data_cleaner,                
                age_group_categorizer,       
                income_group_categorizer,   
                loan_amount_categorizer,     
                ratio_feature_generator,           
            ]

            
            for im_one_hot_feature, string_indexer_col in zip(self.schema.required_oneHot_features(),
                                                              self.schema.string_indexer_one_hot_features):
                string_indexer = StringIndexer(inputCol=im_one_hot_feature, outputCol=string_indexer_col)
                stages.append(string_indexer)
            onehot_encoded = OneHotEncoder(inputCols=self.schema.string_indexer_one_hot_features,
                                                   outputCols=self.schema.output_one_hot_encoded_feature)
            stages.append(onehot_encoded)

            
        
            

            min_max_scaler = MinMaxScaler(inputCol=self.schema.output_assambling_column,outputCol=self.schema.min_max_features)

            assembler = VectorAssembler(inputCols=self.schema.assambling_columns, outputCol=self.schema.output_assambling_column)
            stages.append(customTransformer)
            stages.append(assembler)
            stages.append(min_max_scaler)
        
            
             
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
            print("before pipeline")
     

            pipeline = self.get_data_transformation_pipeline()
            transformed_pipeline = pipeline.fit(train_dataframe)
            transformed_trained_dataframe = transformed_pipeline.transform(train_dataframe)
            print("after  pipeline")
            transformed_trained_dataframe.printSchema()


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
    



