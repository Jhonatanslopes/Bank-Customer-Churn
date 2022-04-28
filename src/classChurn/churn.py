import pickle
import numpy as np
import pandas as pd
import json
import sklearn
import inflection

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


class Churn:
    def __init__(self):
        
        # LOCAL:
        #self.path = 'C:/Users/Jhonatans/projects/ML/Classification/Bank-Customer-Churn/src/'

        ## load transformation saved as object
        #self.mms_balance = pickle.load(open(str(self.path) + 'preparation/mms_balance.pkl', 'rb'))
        #self.mms_salary = pickle.load(open(str(self.path) + 'preparation/mms_salary.pkl', 'rb'))
        #self.rs_age = pickle.load(open(str(self.path) + 'preparation/rs_age.pkl', 'rb'))
        #self.rs_customer_return = pickle.load(open(str(self.path) + 'preparation/rs_customer_return.pkl', 'rb'))
        #self.sc_credit_score = pickle.load(open(str(self.path) + 'preparation/sc_credit_score.pkl', 'rb'))

        # WEB/CLOUD:
        ## load transformation saved as object
        self.mms_balance = pickle.load(open('preparation/mms_balance.pkl', 'rb'))
        self.mms_salary = pickle.load(open('preparation/mms_salary.pkl', 'rb'))
        self.rs_age = pickle.load(open('preparation/rs_age.pkl', 'rb'))
        self.rs_customer_return = pickle.load(open('preparation/rs_customer_return.pkl', 'rb'))
        self.sc_credit_score = pickle.load(open('preparation/sc_credit_score.pkl', 'rb'))

    def cleaning(self, df):

        """
            recebe o dado cru e realiza as limpezas.

            df = dado cru (dataframe pandas)
        """

        # rename columns to snakecase
        snakecase = lambda x: inflection.underscore(x)
        new_columns = (map(snakecase, df.columns))
        df.columns = new_columns

        return df


    def feature_engineering(self, df):

        """
            recebe o dado limpo e rezalia a criação de mais 
            features com aquelas já existentes.
            
            df = dado limpo (dataframe pandas)
        """

        # create feature to score category
        df['credit_score_category'] = df['credit_score'].apply(lambda x: 'ruim' if x > 300 and x <= 500 else 'regular' if x > 500 and x <= 700 else 'bom')

        # create feature to age category
        df['category_age'] = df['age'].apply(lambda x: '>= 40' if x > 40 else '< 40')

        # create feature to salary category
        MEDIAN_SALARY = df['estimated_salary'].median()
        df['category_salary'] = df['estimated_salary'].apply(
            lambda x: 'higher salary' if x > MEDIAN_SALARY else 'lower salary')
        
        # create feature to num of products
        df['num_of_products_category'] = df['num_of_products'].apply(
            lambda x: '< 2' if x < 2 else '>= 2')

        # create feature to tenure category
        df['tenure_category'] = df['tenure'].apply(lambda x: '< 3' if x < 3 else '>= 3')

        # create feature to balance zero
        df['balance_zero'] = df['balance'].apply(lambda x: 'zero' if x == 0.0 else 'not zero')

        # create feauture to balance category
        MEDIAN_BALANCE = df['balance'].median()
        df['category_balance'] = df['balance'].apply(lambda x: 'higher balance' if x > MEDIAN_BALANCE else 'lower balance')

        # create feature to aquisition power
        df['purchasing_power'] = df['estimated_salary'].apply(
            lambda x: 'low' if x < 3000 else 'regular'
                            if x >= 3000 and x < 8000 else 'high')

        # create feature to customer return
        avg_salary = df['estimated_salary'].mean()
        df['customer_return'] = df[['estimated_salary', 'tenure']].apply(
            lambda x: (x['estimated_salary'] * 20) / 100 * x['tenure'] if x['estimated_salary'] > avg_salary else (x['estimated_salary'] * 15) / 100 * x['tenure'], axis=1)

        return df

    
    def preparation(self, df):

        """
            recebe o dado limpo e com as features criadas 
            e prepara o dado (enconding/reescaling) para a predição.

            df = dado com engenharia de feauture criada. (dataframe pandas)
        """

        # RESCALING
        ## apply min max scaler
        df['balance'] = self.mms_balance.transform(df[['balance']].values)
        df['estimated_salary'] = self.mms_salary.transform(df[['estimated_salary']].values) 

        ## apply robust scaler
        df['age'] = self.rs_age.transform(df[['age']].values)
        df['customer_return'] = self.rs_customer_return.transform(df[['customer_return']].values)

        ## apply standard scaler
        df['credit_score'] = self.sc_credit_score.transform(df[['credit_score']].values)

        # ENCODING
        ## apply one hot encoding
        df['geography_Germany'] = df['geography'].apply(lambda x: 1 if x == 'Germany' else 0)
        df['geography_France'] = df['geography'].apply(lambda x: 1 if x == 'France' else 0)
        df['geography_Spain'] = df['geography'].apply(lambda x: 1 if x == 'Spain' else 0)
        df['purchasing_power_low'] = df['purchasing_power'].apply(lambda x: 1 if x == 'low' else 0)
        df['purchasing_power_regular'] = df['purchasing_power'].apply(lambda x: 1 if x == 'regular' else 0)
        df['purchasing_power_high'] = df['purchasing_power'].apply(lambda x: 1 if x == 'high' else 0)
    
        ## apply binary encoding
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Female' else 0)
        df['category_age'] = df['category_age'].map({'>= 40': 1, '< 40': 0})
        df['category_salary'] = df['category_salary'].map({'lower salary': 0, 'higher salary': 1})
        df['num_of_products_category'] = df['num_of_products_category'].map({'< 2': 0, '>= 2': 1})
        df['tenure_category'] = df['tenure_category'].map({'< 3': 0, '>= 3': 1})
        df['balance_zero'] = df['balance_zero'].map({'zero': 0, 'not zero': 1})
        df['category_balance'] = df['category_balance'].map({'lower balance': 0, 'higher balance': 1})

        ## apply label encoding
        df['credit_score_category'] = df['credit_score_category'].map({'ruim': 0, 'regular': 1, 'bom': 2})

        return df


    def feature_selection(self, df):

        """
            recebe o dados limpo, com features criadas, com as prearações
            feitas e filtra as features mais importrantes para a predição do modelo. 

            df = dado preparado. (dataframe pandas)
        """

        cols_selected_importance = ['age', 'credit_score', 'estimated_salary', 'num_of_products', 'customer_return',
                                    'category_age', 'balance', 'tenure', 'is_active_member', 'credit_score_category',
                                    'num_of_products_category', 'has_cr_card', 'gender']

        cols_selected_boruta = ['age', 'balance', 'is_active_member', 'category_age', 'num_of_products_category', 'num_of_products']
        cols_selected_final = list(set(cols_selected_importance + cols_selected_boruta))

        df_filtered = df[cols_selected_final]

        return df_filtered
    
    def get_prediction(self, model, original_data, test_data):

        """
            recebe o instancia do modelo treinado, dado cru, dado processado e
                realiza a predição ajuntando com o dado cru.

                model = intancia/objeto do modelo treinado 
                original_data = dado cru (dataframe pandas).
                test_data = dado processado (dataframe pandas).
        """
        
        # model prediction
        pred = model.predict_proba(test_data.values)

        # join prediction into original data
        original_data['churn_prediction'] = list(pred[:, 1])

        # original data with predictino to json
        return original_data.to_json(orient='records', date_format='iso')