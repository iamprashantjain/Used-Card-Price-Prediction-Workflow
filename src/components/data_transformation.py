import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation.

    Attributes:
        preprocessor_obj_file_path (str): The file path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    """
    This class handles the data transformation processes required for training and testing machine learning models.
    
    Methods:
        get_data_preprocessing: Creates a preprocessing pipeline for numerical and categorical features.
        initialize_data_transformation: Reads training and testing data, applies preprocessing, and saves the preprocessor.
    """
    
    def __init__(self):
        """
        Initializes the DataTransformation class and its configuration.
        """
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_preprocessing(self):
        """
        Sets up the preprocessing pipeline for data transformation.

        The preprocessing consists of separate pipelines for numerical and categorical features:
        - Numerical features are imputed using the median and then scaled.
        - Categorical features are imputed using the most frequent value, ordinal encoded, and then scaled.

        Returns:
            ColumnTransformer: A transformer that applies the appropriate transformations to the specified columns.
        """
        try:
            logging.info('Preprocessing initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            
            logging.info('Pipeline Initiated')
            
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Combined Preprocessing Pipeline
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occurred in get_data_preprocessing")
            raise customexception(e, sys)
    
    def initialize_data_transformation(self, train_path: str, test_path: str):
        """
        Reads the training and testing datasets, applies the preprocessing pipeline, and saves the preprocessor object.

        Args:
            train_path (str): The file path for the training dataset.
            test_path (str): The file path for the testing dataset.

        Returns:
            tuple: A tuple containing:
                - train_arr (ndarray): Processed training data with target variable.
                - test_arr (ndarray): Processed testing data with target variable.
        """
        try:
            # Load the datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            # Get the preprocessing object
            preprocessing_obj = self.get_data_preprocessing()
            
            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']
            
            # Prepare the features and target variable
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.") 
              
            # concatingnating input features and target variable into new array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object to disk
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)
        
        

# if __name__ == "__main__":
#     # Specify the paths for the training and testing datasets
#     train_data_path = r"H:\CampusX_DS\week43 - My Projects Aug 2024\used_car_price_prediction\Used-Card-Price-Prediction-Workflow\artifacts\train.csv"
#     test_data_path = r"H:\CampusX_DS\week43 - My Projects Aug 2024\used_car_price_prediction\Used-Card-Price-Prediction-Workflow\artifacts\test.csv"

#     # Create an instance of the DataTransformation class
#     data_transformation = DataTransformation()
    
#     # Call the initialize_data_transformation method
#     try:
#         train_array, test_array = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
#         print("Training and testing data transformation completed successfully.")
#         print("Training array shape:", train_array.shape)
#         print("Testing array shape:", test_array.shape)

#     except Exception as e:
#         print("An error occurred during data transformation:", e)
