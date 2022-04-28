# imports
import pickle
import pandas as pd
import os
import sklearn
import numpy as np
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from flask import Flask, request, Response
from classChurn.churn import Churn


model = pickle.load(open('/model/model.pkl', 'rb')) # load model saved with pickle
app = Flask(__name__) # initialize API

@app.route('/CustomerChurn/predict', methods=['POST'])
def customer_churn():

    test_json = request.get_json()
    if test_json: # there is data

        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])

        else: # multiple exemples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instance class
        churn = Churn()

        # data cleaning
        df_cleaning = churn.cleaning(df=test_raw)
        print('cleaning OK')

        # feature engineering
        df_feature = churn.feature_engineering(df=df_cleaning)
        print('feature engineering OK')

        # data preparation
        df_prearation = churn.preparation(df=df_feature)
        print('prearation OK')

        # feature selection
        df_filtered = churn.feature_selection(df=df_prearation)
        print('feature selection OK')

        # prediction
        df_response = churn.get_prediction(model=model, original_data=df_cleaning, test_data=df_filtered)
        print('prediction OK')

        return df_response 


if __name__ == '__main__':
    porta = os.environ.get('PORT', 5000)
    app.run(host='127.0.0.1', port=porta)