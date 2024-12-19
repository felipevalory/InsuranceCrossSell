import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from API.health_insurance.health_insurance import HealthInsurance

# loading model
home_path = os.getcwd()
model = (pickle.load(open(os.path.join(
    home_path,
    'src',
    'models',
    'model_health_insurance_cross_sell.pkl', 'rb'))))

# initialize API
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json:  # there is data
        if isinstance(test_json, dict):  # unique exemple
            test_raw = pd.DataFrame(test_json, index=[0])
        else:  # multiple exemples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instantiate HealthInsurance class
        pipeline = HealthInsurance()

        # Data cleaning
        data = pipeline.rename_columns(test_raw)

        # Feature Engineering
        data = pipeline.feature_engineering(data)

        # Data Preparation
        data = pipeline.data_preparation(data)

        # Prediction
        data_response = pipeline.get_prediction(model, test_raw, data)

        return data_response

    else:
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port)
