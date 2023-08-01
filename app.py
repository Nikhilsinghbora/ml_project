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
    if Request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=Request.form.get('gender'),
            race_ethnicity=Request.form.get('race_ethinicity'),
            parental_level_of_education=Request.form.get(
                'parental_level_of_education'),
            lunch=Request.form.get('lunch'),
            test_preparation_course=Request.form.get('writing_score'),
            writing_score=float(Request.form.get('reading_score')),
            reading_score=float(Request.form.get('reading_score')),
        )

        pread_df = data.get_data_as_data_frame()
        print(pread_df)

        predict_pipeline = Predictpipeline()
        result = predict_pipeline.predict(pread_df)

        return render_template('home.html', results=result[0])


if __name__ == "main":
    app.run(debug=True)