from flask import Flask,request,render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender= request.form.get('gender'),
            ethnicity= request.form.get('ethnicity'),
            parental_education= request.form.get('parental_education'),
            lunch= request.form.get('lunch'),
            test_prep_course= request.form.get('test_prep_course'),
            reading_score= float(request.form.get('reading_score')),
            writing_score= float(request.form.get('writing_score')),
        )
        ip_df = data.get_as_dataframe()
        print(ip_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(ip_df)
        return render_template('home.html',results=int(pred[0]))

if __name__=="__main__":
    app.run(host='0.0.0.0')