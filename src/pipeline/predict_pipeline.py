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
    def __init__(self, insured_zip, collision_type, incident_severity, authorities_contacted, insured_sex,
                 capital_gains, capital_loss, insured_relationship, policy_state, umbrella_limit, incident_type,
                 insured_education_level, policy_deductable, policy_annual_premium):
        self.insured_zip = insured_zip
        self.collision_type = collision_type
        self.incident_severity = incident_severity
        self.authorities_contacted = authorities_contacted
        self.insured_sex = insured_sex
        self.capital_gains = capital_gains
        self.capital_loss = capital_loss
        self.insured_relationship = insured_relationship
        self.policy_state = policy_state
        self.umbrella_limit = umbrella_limit
        self.incident_type = incident_type
        self.insured_education_level = insured_education_level
        self.policy_deductable = policy_deductable
        self.policy_annual_premium = policy_annual_premium

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "insured_zip": [self.insured_zip],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "insured_sex": [self.insured_sex],
                "capital_gains": [self.capital_gains],
                "capital_loss": [self.capital_loss],
                "insured_relationship": [self.insured_relationship],
                "policy_state": [self.policy_state],
                "umbrella_limit": [self.umbrella_limit],
                "incident_type": [self.incident_type],
                "insured_education_level": [self.insured_education_level],
                "policy_deductable": [self.policy_deductable],
                "policy_annual_premium": [self.policy_annual_premium],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(str(e), sys)
