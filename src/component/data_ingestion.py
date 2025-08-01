import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.component.data_transformation import DataTransformation,DataTransformationConfig
from src.component.model_trainer import ModelTrainer,ModelTrainerConfig
from src.component.hyper_params_tuning import HyperParamTuning,HyperParamTuningConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Entered in the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\Data\stud.csv')
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Directory Created and data frame saved.")
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Inmgestion of the data is completed")
            return (self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_set,test_set = obj.initiate_data_ingestion()
    transformation_obj = DataTransformation()
    train_arr,test_arr,_ = transformation_obj.initiate_transformation(train_set,test_set)
    model_trainer_obj = ModelTrainer()
    _,__,___=model_trainer_obj.initiate_model_trainer(train_arr,test_arr)
    hyper_param_obj = HyperParamTuning()
    _,__,___ = hyper_param_obj.initiate_hyper_param_tuning(train_arr,test_arr)
    logging.info("Code run Successfully")