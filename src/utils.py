import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle

def save_object(filepath,obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as f:
            pickle.dump(obj,f)
        
    except Exception as e:
        raise CustomException(e,sys)