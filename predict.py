from credit_risk.pipeline.prediction import Prediction
from credit_risk.entity.config_entity import BatchPredictionConfig

if __name__=="__main__":
    batch_config = BatchPredictionConfig()
    batch_pred = Prediction(batch_config=batch_config)
    batch_pred.start_prediction()