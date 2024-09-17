from datetime import datetime
from credit_risk.entity import DataIngestionConfig,DataIngestionArtifact
from credit_risk.entity.config_entity import DataValidationConfig
from credit_risk.components.data_validation import DataValidation
from credit_risk.entity.config_entity import  TrainingPipelineConfig
from credit_risk.exception import CreditRiskException
from credit_risk.components.data_ingestion import DataIngestion
import os
from credit_risk.pipeline.training import TrainingPipeline

training_pipeline_config = TrainingPipelineConfig()
tr = TrainingPipeline(training_pipeline_config=training_pipeline_config)
tr.start() 
