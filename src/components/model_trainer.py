import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# =========================================================
# Model Trainer Config
# =========================================================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# =========================================================
# Model Trainer
# =========================================================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                },
                "Random Forest": {
                    "n_estimators": [32, 64, 128],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.05],
                    "n_estimators": [64, 128],
                },
            }

            logging.info("Evaluating models")

            model_report, best_model, best_model_score = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            logging.info(f"Model performance: {model_report}")
            logging.info(f"Best model score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
