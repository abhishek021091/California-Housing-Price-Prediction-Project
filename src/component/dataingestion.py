import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.component.datacleaning import DataCleaningConfig, DataCleaner

from src.component.datatransformation import DataTransformationConfig, DataTransformer

from src.component.modeltraining import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact', "train.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("E:\California Housing Price Prediction Project\California-Housing-Price-Prediction-Project\data\housing.csv")
            logging.info('Read the datasets as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            

            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(df,
                self.ingestion_config.train_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,_=obj.initiate_data_ingestion()

    cleaner = DataCleaner()
    _,train_cleaned_path= cleaner.clean_training_data(train_data)

    data_tranformation=DataTransformer()
    train_arr,_=data_tranformation.initiate_data_transformation(train_cleaned_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr))
