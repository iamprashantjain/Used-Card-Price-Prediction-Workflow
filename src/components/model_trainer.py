import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for Model Trainer.

    Attributes:
        trained_model_file_path (str): The file path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    This class is responsible for training various regression models on the provided training data
    and selecting the best model based on performance.

    Methods:
        initate_model_training: Trains multiple regression models and identifies the best performing one.
    """
    
    def __init__(self):
        """
        Initializes the ModelTrainer class and its configuration.
        """
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array: np.ndarray, test_array: np.ndarray):
        """
        Trains regression models on the provided training data and evaluates their performance.

        Args:
            train_array (ndarray): The training data, with features and target variable.
            test_array (ndarray): The testing data, with features and target variable.

        Returns:
            None: The method saves the best performing model to a file.

        Raises:
            customexception: If an error occurs during model training.
        """
        try:
            logging.info('Splitting dependent and independent variables from train and test data')

            # Split the training and testing data into features and target variable
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features from training data
                train_array[:, -1],   # Target from training data
                test_array[:, :-1],   # Features from testing data
                test_array[:, -1]     # Target from testing data
            )

            # Dictionary of models to evaluate
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }
            
            # Evaluate models and get performance report
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            # Identify the best model
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred during model training')
            raise customexception(e, sys)