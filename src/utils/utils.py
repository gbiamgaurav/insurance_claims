import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

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
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i, model_name in enumerate(models):
            model = models[model_name]
            parameters = params[model_name]
            gs = GridSearchCV(model, parameters, cv=3)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)  # Fit the model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)