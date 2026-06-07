import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

"""from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig """  

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading score', 'writing score']  
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
       'test preparation course'] 

            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Numerical and categorical pipelines are created")

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', numerical_pipeline, numerical_features),
                ('cat_pipeline', categorical_pipeline, categorical_features)
            ])
        
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read the train and test dataset as dataframe")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

           
            target_column_name = 'math score'  # Replace with your actual target column name
            numerical_features = [ 'reading score', 'writing score']

            y_train = train_df[target_column_name]
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            
            y_test = test_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            
            print(X_train.columns)
            logging.info("applying preprocessing object on training and testing dataframe")


           
            X_train_preprocessed = preprocessing_obj.fit_transform(X_train)
            X_test_preprocessed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_preprocessed, np.array(y_train)]
            test_arr = np.c_[X_test_preprocessed, np.array(y_test)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj 
            )

            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                    )

        except Exception as e:
                raise CustomException(e, sys)
        


