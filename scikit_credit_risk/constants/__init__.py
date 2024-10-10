"""
author:@ kumar dahal
"""
import os
from datetime import datetime
import numpy as np

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
ALL_COLUMNS = 'columns'
    
ROOT_DIR = os.getcwd()  #to get current working directory
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

MODDEL_PARAMS = "model_params.yaml"
MODDEL_PARAMS_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,MODDEL_PARAMS)

CURRENT_TIME_STAMP = get_current_time_stamp()



# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"


# Data Ingestion related variable

DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY = "tgz_download_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

# Data Validation related variable

# Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_ARTIFACT_DIR_NAME="data_validation"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"


# Data Transformation related variables
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_LOAN_TO_INCOME_RATIO = "loan_to_income_ratio"
DATA_TRANSFORMATION_LOAN_TO_EMP_LENGTH_RATIO = "loan_to_emp_length_ratio"
DATA_TRANSFORMATION_INT_RATIO_TO_LOAN_AMT_RATIO = "int_rate_to_loan_amt_ratio"
DATA_TRANSFORMATION_INCOME_GROUP = "income_group"
DATA_TRANSFORMATION_AGE_GROUP = "age_group"
DATA_TRANSFORMATION_LOAN_AMOUNT_GROUP = "loan_amount_group"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_object_file_name"


DATASET_SCHEMA_COLUMNS_KEY=  "columns"

NUMERICAL_COLUMN_KEY="numerical_columns"
CATEGORICAL_COLUMN_KEY = "categorical_columns"


TARGET_COLUMN_KEY="target_column"


# Model Training related variables

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"

#for model evaluation
MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_ARTIFACT_DIR = "model_evaluation"

#contain of model_evaluation.yaml
BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

# Model Pusher config key for  saving models and we are  saving model in main directory  saved_models
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"



EXPERIMENT_DIR_NAME="experiment"
EXPERIMENT_FILE_NAME="experiment.csv"



N_ESTIMATORS= [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]  # number of trees in the random forest
MAX_DEPTH =  [int(x) for x in np.linspace(5, 30, num = 6)]  # maximum number of levels allowed in each decision tree
MIN_SAMPLES_SPLIT =   [2, 5, 10, 15, 100]  # minimum sample number to split a node
MIN_SAMPLES_LEAF =  [1, 2, 5, 10]  # minimum sample number that can be stored in a leaf node
BOOTSTRAP =  [True, False]  # method used to sample data points
MAX_LEAF_NODES = range(3,9,1)

AGE_BINS = [20, 26, 36, 46, 56, 66, 80]
AGE_LABELS = ['20_25', '26_35', '36_45', '46_55', '56_65', '66_80']

# @dataclass
# class EnvironmentVariable:
#     mongo_db_url = os.getenv("MONGO_DB_URL")


# env_var = EnvironmentVariable()


#model factory
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"