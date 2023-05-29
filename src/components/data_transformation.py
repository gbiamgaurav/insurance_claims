import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from src.utils.utils import save_object
from src.exception import CustomException
from src.logger import logging
from kneed import KneeLocator

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
                    ("robustscaler", RobustScaler()),
                    ("pca", PCA(n_components=len(numerical_columns)))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="drop"
            )

            return preprocessor
            logging.info("Preprocessor object obtained")
        except Exception as e:
            raise CustomException("Error occurred while obtaining the preprocessor object", sys)

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

            # Perform PCA to determine the optimal number of components
            pca = preprocessing_obj.named_transformers_['num_pipeline']['pca']
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = KneeLocator(range(1, len(explained_variance) + 1), explained_variance, curve="convex").knee

            # Update the number of components in the PCA transformer
            preprocessor.named_transformers_['num_pipeline']['pca'].n_components = n_components

            logging.info("Applying PCA")
            
            # Transform the data
            train_transformed = preprocessing_obj.transform(input_feature_train_df)
            test_transformed = preprocessing_obj.transform(input_feature_test_df)

            # Save the preprocessor object for later use
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            # Select the most important features based on PCA component weights
            pca_component_weights = pca.components_
            feature_importance = np.abs(pca_component_weights).sum(axis=0)
            selected_features_indices = np.argsort(feature_importance)[::-1]  # Sort in descending order
            selected_features = input_feature_train_df.columns[selected_features_indices][:n_components]

            logging.info("The selected features are: ")

            print("Selected Features:")
            for feature in selected_features:
                print(feature)


            return (
                train_transformed,
                test_transformed,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            logging.info("Saved preprocessor object")

        except Exception as e:
            raise CustomException("Error occurred during feature transformation", sys)
