from datetime import datetime
import os
from dataclasses import dataclass
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

AGE_BINS = [20, 26, 36, 46, 56, 66, 80]
AGE_LABELS = ['20_25', '26_35', '36_45', '46_55', '56_65', '66_80']

@dataclass
class EnvironmentVariable:
    mongo_db_url = os.getenv("MONGO_DB_URL")


env_var = EnvironmentVariable()