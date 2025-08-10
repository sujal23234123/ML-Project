import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
         
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(x_train, y_train, x_test, y_test, models,params):
    try:
        report = {}

        for model_name, model in models.items():
            # Correctly fetching parameters using the model_name key
            para = params[model_name] 

            # GridSearchCV to find the best parameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)

            # Setting the model with the best parameters found
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # Making predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calculating R-squared scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)