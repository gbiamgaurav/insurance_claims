import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            model = load_object(file_path=model_path)
            pred = model.predict(features)
            return pred
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 insured_sex: str,
                 incident_type: str,
                 collision_type: str,
                 incident_severity: str,
                 incident_city: str,
                 number_of_vehicles_involved: int,
                 property_damage: str,
                 police_report_available: str,
                 bodily_injuries: int):
        
        self.insured_sex = insured_sex
        self.incident_type = incident_type
        self.collision_type = collision_type
        self.incident_severity = incident_severity
        self.incident_city = incident_city
        self.number_of_vehicles_involved = number_of_vehicles_involved
        self.property_damage = property_damage
        self.police_report_available = police_report_available
        self.bodily_injuries = bodily_injuries

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'insured_sex': [self.insured_sex],
                'incident_type': [self.incident_type],
                'collision_type': [self.collision_type],
                'incident_severity': [self.incident_severity],
                'incident_city': [self.incident_city],
                'number_of_vehicles_involved': [self.number_of_vehicles_involved],
                'property_damage': [self.property_damage],
                'police_report_available': [self.police_report_available],
                'bodily_injuries': [self.bodily_injuries],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
