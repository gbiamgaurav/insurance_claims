import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from xgboost import XGBClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object, load_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForest": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            p_score = precision_score(y_test, predicted)

            return acc_score, p_score

        except Exception as e:
            raise CustomException(e, sys)
