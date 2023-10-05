import os
import sys
from dataclasses import dataclass


import csv
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        try:

            logging.info("train and test split")

            x_train,y_train,x_test,y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                     "Linear Regression": LinearRegression(),
                     "Lasso": Lasso(),
                     "Ridge": Ridge(),
                     "Gradient Boosting": GradientBoostingRegressor(),
                     "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                     "AdaBoost Regressor": AdaBoostRegressor(),
                     "K-Neighbors" : KNeighborsRegressor(),
                     }

            """
            for applying hyperparameter tuning apply u can add another parameter to
            evaluate_models that is dictionary of grid of all hyperparameters
            then u use grid/random search cv in utils.py i.e by changing
            """
            
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,
            y_test=y_test,models=models)

            
 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6 :

                raise CustomException("No best model found")

            logging.info(f"best model found in our data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted= best_model.predict(x_test)

            r2_sco = r2_score(y_test,predicted)

            return r2_sco

        except Exception as e:

            raise CustomException(e,sys)


            