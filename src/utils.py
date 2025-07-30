import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import pickle

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as f:
            pickle.dump(obj,f)
        
    except Exception as e:
        raise CustomException(e,sys)
    

def evalute_models(x_train,x_test,y_train,y_test,models):
    try:
        report=dict()

        for name,model in models.items():
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_score = r2_score(y_train,y_train_pred)
            test_score = r2_score(y_test,y_test_pred)
            report[name]=(test_score,model)
        sorted_report = dict(sorted(report.items(),key=lambda item:item[1][0],reverse=True))
        return sorted_report

    except Exception as e:
        raise CustomException(e,sys)
    
    
def hyper_param_tuning(x_train,x_test,y_train,y_test,models):
    try:
        models_best_params = {}
        for name,tup in models.items():
            model=tup[0]
            params = tup[1]
            random = RandomizedSearchCV(estimator=model,param_distributions=params,
                        scoring='neg_mean_squared_error',cv=5,random_state=42,n_jobs=-1)
            random.fit(x_train,y_train)
            best_params = random.best_params_
            pred= random.predict(x_test)
            score = r2_score(y_test,pred)
            models_best_params[name]=(score,best_params)
        return dict(sorted(models_best_params.items(),key=lambda item:item[1][0],reverse=True))
    
    except Exception as  e:
        raise CustomException(e,sys)
    

def load_object(filepath):
    try:
        with open(filepath,'rb') as f:
            obj = pickle.load(f)
            return obj
    except Exception as e:
        raise CustomException(e,sys)