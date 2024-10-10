from datetime import datetime

from scikit_credit_risk.entity.config_entity import DataValidationConfig
from scikit_credit_risk.components.data_validation import DataValidation
from scikit_credit_risk.exception import CreditException
from scikit_credit_risk.config.configuration import Configuartion
from scikit_credit_risk.components.data_ingestion import DataIngestion
import os
from scikit_credit_risk.pipeline.pipeline import Pipeline
from scikit_credit_risk.constants import get_current_time_stamp

pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
else:
    message = "Training is already in progress."
    print(message)
