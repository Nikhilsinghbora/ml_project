import flask
from flask import Flask, Request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, Predictpipeline

application = Flask(__name__)

app = application

# Route for a homepage


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    req = flask.request
    if req.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=req.form.get('gender'),
            race_ethnicity=req.form.get('race_ethnicity'),
            parental_level_of_education=req.form.get(
                'parental_level_of_education'),
            lunch=req.form.get('lunch'),
            test_preparation_course=req.form.get('test_preparation_course'),
            writing_score=float(req.form.get('writing_score')),
            reading_score=float(req.form.get('reading_score')),
        )
        # Request.form

        pread_df = data.get_data_as_data_frame()
        print(pread_df)

        predict_pipeline = Predictpipeline()
        result = predict_pipeline.predict(pread_df)

        return render_template('home.html', results=result[0])


if __name__ == "__main__":
    app.run(port=8000,debug=True)
