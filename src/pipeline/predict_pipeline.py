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
    def __init__(self,insured_sex,insured_relationship,collision_type,
       incident_severity,incident_state,incident_city,
       property_damage,police_report_available,bodily_injuries,witnesses):
        
        self.insured_sex = insured_sex
        self.insured_relationship = insured_relationship
        self.collision_type = collision_type
        self.incident_severity = incident_severity
        self.incident_state = incident_state
        self.incident_city = incident_city
        self.property_damage = property_damage
        self.police_report_available = police_report_available
        self.bodily_injuries = bodily_injuries
        self.witnesses = witnesses

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "insured_sex": [self.insured_sex],
                "insured_relationship": [self.insured_relationship],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "incident_state": [self.incident_state],
                "incident_city": [self.incident_city],
                "property_damage": [self.property_damage],
                "police_report_available": [self.police_report_available],
                "bodily_injuries": [self.bodily_injuries],
                "witnesses": [self.witnesses]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(str(e), sys)
