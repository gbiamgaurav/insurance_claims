import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass 
    
    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            if preprocessor is None:
                raise CustomException("Preprocessor not found", sys)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(str(e), sys)

class CustomData:
    def __init__(self, incident_severity, incident_type,
                 policy_annual_premium):
        
        self.incident_severity = incident_severity
        self.incident_type = incident_type
        self.policy_annual_premium = policy_annual_premium

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "incident_severity": [self.incident_severity],
                "incident_type": [self.incident_type],
                "policy_annual_premium": [self.policy_annual_premium]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(str(e), sys)
