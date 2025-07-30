import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            modelpath='artifacts\model.pkl'
            preprocessorpath='artifacts\preprocessor.pkl'
            model = load_object(modelpath)
            preprocessor = load_object(preprocessorpath)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,gender,ethnicity,parental_education,lunch, test_prep_course,
                reading_score, writing_score):
        self.gender = gender
        self.ethnicity = ethnicity
        self.parental_education = parental_education
        self.lunch = lunch
        self.test_prep_course = test_prep_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_as_dataframe(self):
        try:
            data_input = dict(
                gender= [self.gender],
                ethnicity= [self.ethnicity],
                parental_education= [self.parental_education],
                lunch= [self.lunch],
                test_prep_course= [self.test_prep_course],
                reading_score= [self.reading_score],
                writing_score= [self.writing_score]
            )

            return pd.DataFrame(data_input)
        except Exception as e:
            raise CustomException(e,sys)
        

        
        
