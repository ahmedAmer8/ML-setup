import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = "artifacts/model.pkl"
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and testing arrays into features and target variable")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0)
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predictions = best_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            return r2
            # return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)