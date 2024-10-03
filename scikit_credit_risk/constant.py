from datetime import datetime
import os
from dataclasses import dataclass
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
import numpy as np
AGE_BINS = [20, 26, 36, 46, 56, 66, 80]
AGE_LABELS = ['20_25', '26_35', '36_45', '46_55', '56_65', '66_80']

@dataclass
class EnvironmentVariable:
    mongo_db_url = os.getenv("MONGO_DB_URL")


env_var = EnvironmentVariable()


N_ESTIMATORS= [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]  # number of trees in the random forest
MAX_DEPTH =  [int(x) for x in np.linspace(5, 30, num = 6)]  # maximum number of levels allowed in each decision tree
MIN_SAMPLES_SPLIT =   [2, 5, 10, 15, 100]  # minimum sample number to split a node
MIN_SAMPLES_LEAF =  [1, 2, 5, 10]  # minimum sample number that can be stored in a leaf node
BOOTSTRAP =  [True, False]  # method used to sample data points
MAX_LEAF_NODES = range(3,9,1)

#model factory
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"