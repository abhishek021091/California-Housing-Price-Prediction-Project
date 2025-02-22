import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array):
        try:
            logging.info("Splitting features and target variable")

            X = train_array[:, :-1]  # Features
            y = train_array[:, -1]   # Target

            # Define Random Forest with given parameters
            model = RandomForestRegressor(
                max_depth=1000,
                max_features=0.4606,
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=1000,
                random_state=1,
                n_jobs=-1  # Use all CPU cores for faster training
            )

            logging.info("Training Random Forest on full dataset...")
            model.fit(X, y)  # Train the model on the full dataset

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

            return model  # Return the trained model

        except Exception as e:
            raise CustomException(e, sys)
