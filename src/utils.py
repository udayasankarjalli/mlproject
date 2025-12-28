import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


# =========================================================
# Save object (model / preprocessor)
# =========================================================
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# =========================================================
# Load object
# =========================================================
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# =========================================================
# Evaluate models and return BEST TRAINED MODEL
# =========================================================
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models and returns:
    - performance report
    - best trained model
    - best model score
    """

    try:
        report = {}
        best_model = None
        best_score = float("-inf")

        for model_name, model in models.items():

            # ---------------------------------------------
            # CatBoost (NO GridSearchCV)
            # ---------------------------------------------
            if model_name == "CatBoosting Regressor":
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                score = r2_score(y_test, y_test_pred)

            # ---------------------------------------------
            # sklearn models (WITH GridSearchCV)
            # ---------------------------------------------
            else:
                params = param[model_name]

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)

                model = gs.best_estimator_
                y_test_pred = model.predict(X_test)
                score = r2_score(y_test, y_test_pred)

            report[model_name] = score

            if score > best_score:
                best_score = score
                best_model = model

        return report, best_model, best_score

    except Exception as e:
        raise CustomException(e, sys)
