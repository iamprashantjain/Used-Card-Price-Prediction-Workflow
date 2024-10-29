import os
import sys
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from urllib.parse import urlparse
import mlflow
import numpy as np
import pickle
import pandas as pd
from src.utils import utils
from src.logger.logging import logging
from src.exception import customexception
from dataclasses import dataclass


@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        pass
    
    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)