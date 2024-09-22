import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model_obj in models.items():
            # Get the parameter grid for the current model
            param = params.get(model_name, {})
            
            # Perform GridSearchCV to find the best parameters
            gs = GridSearchCV(estimator=model_obj, param_grid=param, cv=3)
            
            # Fit GridSearchCV with X_train and y_train (not X_test)
            gs.fit(X_train, y_train)
            
            # Set the best parameters to the model and retrain
            model_obj.set_params(**gs.best_params_)
            model_obj.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model_obj.predict(X_train)
            y_test_pred = model_obj.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
        