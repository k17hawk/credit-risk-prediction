from dataclasses import dataclass


@dataclass(frozen=True)
class InitializeModelDetails:
    model_serial_number: str
    model: str
    param_random_search: str
    model_name: str

@dataclass(frozen=True)
class RandomSearchBestModel:
    model_serial_number: str
    best_model: str
    model: str
    best_parameters: str 
    best_score: str


@dataclass(frozen=True)
class MetricInfoArtifact:
    model_name: str
    model_object: str
    test_precision: float
    test_recall: float
    train_accuracy: float
    test_accuracy: float
    model_accuracy: float
    index_number: int

@dataclass
class BestModel:
    model_serial_number: str
    model: str
    best_model: str
    best_parameters:str
    best_score: float


