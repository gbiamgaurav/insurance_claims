import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from imblearn.combine import SMOTETomek
from src.utils.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column_name = "fraud_reported"

    def get_data_transformer_object(self) -> Pipeline:
        """ This function is responsible for data transformation """
        try:
            numerical_columns = ['policy_deductable', 'policy_annual_premium', 'umbrella_limit',
                                'insured_zip', 'capital-gains', 'capital-loss',
                                'incident_hour_of_the_day', 'number_of_vehicles_involved',
                                'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim',
                                'vehicle_claim', 'auto_year']

            categorical_columns = ['policy_state', 'insured_sex', 'insured_education_level',
                                   'insured_relationship', 'incident_type', 'collision_type',
                                   'incident_severity', 'authorities_contacted', 'incident_state',
                                   'incident_city', 'property_damage', 'police_report_available',
                                   'auto_make']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("robustscaler", RobustScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder()),
                    ("robustscaler", RobustScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            logging.info("Preprocessor file obtained")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[self.target_column_name], axis=1)
            target_feature_train_df = train_df[self.target_column_name]

            input_feature_test_df = test_df.drop(columns=[self.target_column_name], axis=1)
            target_feature_test_df = test_df[self.target_column_name]

            preprocessor = self.get_data_transformer_object()
            preprocessing_obj = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessing_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessing_obj.transform(input_feature_test_df)

            smt = SMOTETomek(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            
            logging.info("Applying preprocessing object on train and test dataframes")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Saved preprocessing object")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)