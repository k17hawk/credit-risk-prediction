
from dataclasses import dataclass
from datetime import datetime
@dataclass
class DataIngestionArtifact:
    download_dir:str
    feature_store_file_path:str

@dataclass
class DataValidationArtifact:
    accepted_file_path:str
    rejected_dir:str