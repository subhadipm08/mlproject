import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evalute_models

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Spliting input and target feature")
            x_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            models = {
            "Linear Regression": LinearRegression(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting Regressor":GradientBoostingRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            model_report = evalute_models(x_train,x_test,y_train,y_test,models)
            best_model_name = list(model_report.keys())[0]
            best_model = model_report[best_model_name][1]
            best_score = model_report[best_model_name][0]

            if best_score<0.6:
                logging.info("All models performance is less than 0.6")
                raise CustomException("No best model Found")
            logging.info("Model Trainning Successful and best model found")
            save_object(
                self.model_trainer_config.trained_model_filepath,
                best_model
            )
            logging.info("model.pkl file created.")
            return (best_model_name, best_model, best_score)

        except Exception as e:
            raise CustomException(e,sys)
