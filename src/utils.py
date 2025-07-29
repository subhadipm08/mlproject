import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
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