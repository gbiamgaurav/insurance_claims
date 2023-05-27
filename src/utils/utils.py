import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False
    import pickle


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        if has_joblib:
            joblib.dump(obj, file_path)
        else:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        if has_joblib:
            obj = joblib.load(file_path)
        else:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)