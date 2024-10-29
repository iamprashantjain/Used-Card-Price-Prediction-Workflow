import os
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# Create an instance of the DataIngestion class
obj = DataIngestion()

# Initiate data ingestion and retrieve paths for training and testing data
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Create an instance of the DataTransformation class
data_transformation = DataTransformation()

# Transform the data and obtain training and testing arrays
train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

# Create an instance of the ModelTrainer class
model_trainer_obj = ModelTrainer()

# Initiate model training using the transformed training and testing data
model_trainer_obj.initate_model_training(train_arr, test_arr)