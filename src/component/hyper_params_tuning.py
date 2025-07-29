import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,hyper_param_tuning

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class HyperParamTuningConfig:
    hyper_model_filepath=os.path.join('artifacts','hypermodel.pkl')


class HyperParamTuning:
    def __init__(self):
        self.hyper_param_config = HyperParamTuningConfig()
    
    def initiate_hyper_param_tuning(self,train_arr,test_arr):
        try:
            logging.info("Spliting input and target feature")
            x_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            models = {
                "Linear Regression": (
                    LinearRegression(),
                    {
                        "fit_intercept": [True, False],
                        "positive": [True, False]
                    }
                ),
                "K-Neighbors Regressor": (
                    KNeighborsRegressor(),
                    {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree"]
                    }
                ),
                "Decision Tree": (
                    DecisionTreeRegressor(),
                    {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                        "max_depth": [5, 10, 15, 20], 
                        "min_samples_split": [5, 8, 10, 15]  
                    }
                ),
                "Gradient Boosting Regressor": (
                    GradientBoostingRegressor(),
                    {   
                        'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                        "n_estimators": [100, 200, 300],
                        "criterion": ["squared_error", "friedman_mse"],
                        "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                        "max_depth": [3, 5, 7 , 10,15]
                    }
                ),
                "Random Forest Regressor": (
                    RandomForestRegressor(),
                    {   
                        "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                        "n_estimators": [100, 200, 300],
                        "max_depth": [5,10,15,20],
                        "min_samples_split": [5, 8, 10, 15]
                    }
                ),
                "XGBRegressor": (
                    XGBRegressor(),
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1, 0.2, 0.5],  
                        "max_depth": [3, 6, 10]
                    }
                ),
                "CatBoosting Regressor": (
                    CatBoostRegressor(verbose=False),
                    {
                        "iterations": [100, 200],
                        "learning_rate": [0.01, 0.1, 0.5], 
                        "depth": [4, 6, 10]
                    }
                ),
                "AdaBoost Regressor": (
                    AdaBoostRegressor(),
                    {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0, 0.5] 
                    }
                )
            }

            models_best_params = hyper_param_tuning(x_train,x_test,y_train,y_test,models)
            logging.info('Hyper Parameter Tuning completed')
            best_hyper_model_score = list(models_best_params.values())[0][0]

            if best_hyper_model_score < 0.6:
                logging.info("All models performance is less than 0.6")
                raise CustomException("No best model Found")
            
            best_hyper_model_name = list(models_best_params.keys())[0]
            best_hyper_model_params = list(models_best_params.values())[0][1]
            best_hyper_model = models[best_hyper_model_name][0]

            best_hyper_model.set_params(**best_hyper_model_params)
            best_hyper_model.fit(x_train,y_train)
            y_pred = best_hyper_model.predict(x_test)
            score = r2_score(y_test,y_pred)

            logging.info("Model Trainning Successful and best model found")
            logging.info(f"Best Model: {best_hyper_model_name},r2_score : {score}")

            save_object(
                self.hyper_param_config.hyper_model_filepath,
                best_hyper_model
            )
            logging.info("hyper_model pkl file created.")
            return (best_hyper_model,best_hyper_model_params,score)

        except Exception as e:
            raise CustomException(e,sys)