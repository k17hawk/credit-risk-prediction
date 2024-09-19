from datetime import datetime
import os
from dataclasses import dataclass
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

AGE_BINS = [20, 26, 36, 46, 56, 66, 80]
AGE_LABELS = ['20-25', '26-35', '36-45', '46-55', '56-65', '66-80']

@dataclass
class EnvironmentVariable:
    mongo_db_url = os.getenv("MONGO_DB_URL")


env_var = EnvironmentVariable()