import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object, evaluate_models
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

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

            logging.info("Evaluating models before hyperparameter tuning")
            report_before_tuning = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info(f"Model evaluation before hyperparameter tuning: {report_before_tuning}")

            param_grid = {
                "RandomForest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10],
                },
                "XGBClassifier": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.1, 0.01, 0.001],
                },
            }

            best_model_score = 0.0
            best_model_name = ""
            best_model = None

            for model_name, model in models.items():
                logging.info(f"Performing hyperparameter tuning for {model_name}")

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid[model_name],
                    scoring="accuracy",
                    cv=5
                )

                grid_search.fit(X_train, y_train)

                if grid_search.best_score_ > best_model_score:
                    best_model_score = grid_search.best_score_
                    best_model_name = model_name
                    best_model = grid_search.best_estimator_

            if best_model is None or best_model_score < 0.7:
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset")

            logging.info(f"Best parameters for {best_model_name}: {best_model.get_params()}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model)

            logging.info("Evaluating best model after hyperparameter tuning")
            report_after_tuning = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models={best_model_name: best_model})
            logging.info(f"Model evaluation after hyperparameter tuning: {report_after_tuning}")

            predicted = best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            p_score = precision_score(y_test, predicted)

            return acc_score, p_score, best_model_name

        except Exception as e:
            raise CustomException(e, sys)