import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_transformer_obj(self):
        try:
            num_cols = ['reading score', 'writing score']
            cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            num_pipe = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )
            cat_pipe = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ohe',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical & numerical col encoder and scaler.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipe,num_cols),
                    ('cat_pipeline',cat_pipe,cat_cols)
                ]
            )
            logging.info("Column transformer is ready.")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read Train and test Data.")

            logging.info("Obtaining preprocessor obj.")
            preprocessor_obj = self.get_transformer_obj()

            target = "math score"
            num_cols = ['reading score', 'writing score']
            cat_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            input_feature_train_df = train_df.drop(target,axis=1)
            target_feature_train_df = train_df[target]
            input_feature_test_df = test_df.drop(target,axis=1)
            target_feature_test_df = test_df[target]

            logging.info("Applying transformation on tarin and test data.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            tarin_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saved Preprocessing obj.")
            save_object(
                self.transformation_config.preprocessor_obj_filepath,
                preprocessor_obj
            )
            return (tarin_arr,test_arr,self.transformation_config.preprocessor_obj_filepath)

        except Exception as e:
            raise CustomException(e,sys)
    
