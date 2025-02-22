import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from shapely.geometry import Polygon
from shapely.geometry import Point
import geopandas as gpd
from haversine import haversine


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")

class DataTransformer:
    
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        
        '''
        This function is responsible for data transformation.

        '''
        try:
            ohe = OneHotEncoder(drop="first",sparse_output=False)
            rbs = RobustScaler()
            std = StandardScaler()
            feature_adder_remover = FeatureAdderRemover()
            
            transformer = ColumnTransformer([
                ("onehot",ohe,['ocean_proximity']),
                ("robust",rbs,['avg_area_to_people','avg_room_area']),
                ("standard",std,['housing_median_age','total_rooms','total_bedrooms','population',
                                        'households','median_income','distance_from_airport','Rooms_per_household',
                                        'people_per_bedroom','bedroom_per_household'])
            ],remainder='passthrough')

            feature_adder_remover = FeatureAdderRemover()
            
            pipeline = Pipeline([
                ("add_remove_feature",feature_adder_remover),
                ("transform",transformer)
            ]) 

            return pipeline
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path):

        try:
            train_df=pd.read_csv(train_path)

            logging.info("Obtaining preprpocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="median_house_value"

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]


            logging.info(
                f"Applying preprocessing object on training dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )


            return (
                train_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )
        

            
        except Exception as e:
            raise CustomException(e,sys)






class FeatureAdderRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        area_dict = {i:geo_area(X_new[X_new['ocean_proximity']==i]) for i in X_new['ocean_proximity'].value_counts().index}
        airport_lon = -122.389977
        airport_lat = 37.615223
        X_new['lon_cos'] = np.cos(X_new['longitude'])
        X_new['lat_sin'] = np.sin(X_new['latitude'])
        X_new['distance_from_airport'] = haversine_df(X_new,'latitude','longitude',airport_lat,airport_lon)
        X_new['bedrooms_rooms_ratio'] = X_new['total_bedrooms']/X_new['total_rooms']
        X_new['Avg_person_per_room']= X_new['population']/X_new['total_rooms']
        X_new['Rooms_per_household'] = X_new['total_rooms']/X_new['households']
        X_new['people_per_bedroom'] = X_new['population']/X_new['total_bedrooms']
        X_new['bedroom_per_household'] = X_new['total_bedrooms']/X_new['households']
        X_new['household_population_ratio'] = X_new['households']/X_new['population']
        X_new['avg_area_to_people'] = X_new['ocean_proximity'].map(area_dict) / (
        X_new['ocean_proximity'].map(X_new['ocean_proximity'].value_counts()) * X_new['population']
        )
        X_new['avg_room_area'] = X_new['ocean_proximity'].map(area_dict)*X_new['total_rooms'] / X_new['ocean_proximity'].map(X_new.groupby(by='ocean_proximity')['total_rooms'].sum())
        X_new.drop(['longitude','latitude'], axis = 1, inplace = True)
        
        return X_new
    
def geo_area(df):
    if len(df) < 4:  # If fewer than 4 points, create a small bounding box
        print("WARNING: Not enough coordinates for polygon, using buffered point.")
        point = Point(df["longitude"].values[0], df["latitude"].values[0])  # Convert to Point
        buffered_polygon = point.buffer(0.01)  # Creates a small artificial area
        gdf = gpd.GeoDataFrame(index=[0], geometry=[buffered_polygon], crs="EPSG:4326")
    else:
        polygon = Polygon(zip(df["longitude"], df["latitude"]))  # Normal polygon
        gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs="EPSG:4326")
    
    gdf = gdf.to_crs(epsg=6933)
    area_m2 = (gdf.area[0]).round(2)
    return area_m2

def haversine_df(df, lat_col, lon_col, ref_lat, ref_lon):
    ref_point = (ref_lat, ref_lon)
    return df.apply(lambda row: haversine(ref_point, (row[lat_col], row[lon_col])), axis=1)