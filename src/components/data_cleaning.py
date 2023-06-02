import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import train_test_split


@dataclass
class DataCleaningConfig:
    train_data_path_cleaned: str = os.path.join("artifacts", "train_cleaned.csv")
    test_data_path_cleaned: str = os.path.join("artifacts", "test_cleaned.csv")

class DataCleaning:
    def __init__(self):
        self.cleaning_config = DataCleaningConfig()

    def initiate_data_cleaning(self):
        logging.info("Entered the data cleaning method or component")
        try:
            df1 = pd.read_csv("artifacts/train.csv")
            df2 = pd.read_csv("artifacts/test.csv")

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path_cleaned), exist_ok=True)
            logging.info("directory created")
            
                        
            logging.info("Dropping some columns")
            
            df1["fraud_reported"].replace({"N": 0, "Y": 1}, inplace=True)
            df2["fraud_reported"].replace({"N": 0, "Y": 1}, inplace=True)
            
            logging.info("Mannual encoding the target column")
            
            logging.info("Data cleaning completed")
            
            df1.to_csv(self.cleaning_config.train_data_path_cleaned,index=False,header=True)
            logging.info('df1 saved as dataframe')

            df2.to_csv(self.cleaning_config.test_data_path_cleaned,index=False,header=True)
            logging.info('df2 saved as dataframe')

            logging.info("Train test data cleaned")

            return df1, df2

            logging.info("returned df1 and df2")

        except Exception as e:
            raise CustomException(e, sys)
