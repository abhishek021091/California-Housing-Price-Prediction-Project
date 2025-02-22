import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataCleaningConfig:
    train_data_path: str=os.path.join('artifact', "train_cleaned.csv")
    

class DataCleaner:
    def __init__(self):
        self.cleaning_config=DataCleaningConfig()
    
    def clean_training_data(self,df):
        """
        Cleans the training dataset by:
        1. Handling missing values using KNN Imputer
        2. Encoding categorical features
        3. Removing extreme outliers (top 10%)
        """
        try:
            logging.info("Data Cleaning Started")
            df_copy = df.copy()
            # Step 1: Handle Missing Values with KNN Imputer
            imputer = KNNImputer(n_neighbors=5)
            
            # Step 2: Encode 'ocean_proximity' with One-Hot Encoding
            ocean_proximity = df_copy[['ocean_proximity']]
            df_copy.drop(columns=['ocean_proximity'], inplace=True)
            
            encoder = OneHotEncoder(sparse_output=False)
            encoded_df = encoder.fit_transform(ocean_proximity)
            encoded_data = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out())
            
            df_encoded = pd.concat([df_copy, encoded_data], axis=1)
            
            # Step 3: Apply KNN Imputer
            df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
            
            # Restore 'total_bedrooms' to original dataframe
            df["total_bedrooms"] = df_imputed["total_bedrooms"].copy()
            
            # Step 4: Remove top 10% extreme values (outliers)
            df = df[df['total_rooms'] < df['total_rooms'].quantile(0.9)]
            df = df[df['total_bedrooms'] < df['total_bedrooms'].quantile(0.9)]
            df = df[df['population'] < df['population'].quantile(0.9)]
            df = df[df['households'] < df['households'].quantile(0.9)]
            df = df[df['median_income'] < df['median_income'].quantile(0.9)]
            
            os.makedirs(os.path.dirname(self.cleaning_config.train_data_path),exist_ok=True)
            df.to_csv(self.cleaning_config.train_data_path,index=False,header=True)
            logging.info("Data Cleaning is completed")

            return (df,
                self.cleaning_config.train_data_path,
            )
        
        except Exception as e:  
            raise CustomException(e,sys)
