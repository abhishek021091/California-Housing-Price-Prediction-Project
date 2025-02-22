import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):

        try:
            features = pd.DataFrame(features, columns=['longitude', 'latitude', 'housing_median_age',
                     'total_rooms', 'total_bedrooms', 'population',
                     'households', 'median_income', 'ocean_proximity'])
            model_path = model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds[0].round(2)
       
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 longitude:float,
                 latitude:float,
                 housing_median_age:float,
                 total_rooms:int,
                 total_bedrooms:int,
                 population:int,
                 households:int,
                 median_income:float,
                 ocean_proximity:str):
        
        self.longitude = longitude 
        self.latitude = latitude
        self.housing_median_age = housing_median_age
        self.total_rooms = total_rooms
        self.total_bedrooms = total_bedrooms
        self.population = population
        self.households = households
        self.median_income = median_income
        self.ocean_proximity = ocean_proximity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "housing_median_age": [self.housing_median_age],
                "total_rooms": [self.total_rooms],
                "total_bedrooms": [self.total_bedrooms],
                "population": [self.population],
                "households": [self.households],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
