import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception


import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    #output path
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        #initiate data ingestion output paths
        self.ingestion_config = DataIngestionConfig()          
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            #read raw data from local folder, cloud, api, dbs etc
            #when reading from cloud or api or dbs we might have to create additional functionality inside utils file to read
            data=pd.read_csv(r"H:\CampusX_DS\week43 - My Projects Aug 2024\used_car_price_prediction\Used-Card-Price-Prediction-Workflow\gemstone.csv")
            logging.info("reading data from source")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("raw data saved in artifact folder")
            
            logging.info("train test split started")            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("train test data saved in artificats")
            logging.info("data ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info()
            raise customexception(e,sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()