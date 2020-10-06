# main.py

import datahandler
import modeler
import pandas as pd

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM

import itertools
from collections import deque

def main():
    
    datahandler.info()
    
    datahandler.corr()

    model_df = pd.read_csv("Xero_features.csv")
    train, test = modeler.tts(model_df)

    # Model 1 - Linear regression
    Linear_Regression = modeler.regressive_model(train, test, LinearRegression(), 'LinearRegression')

    # Model 2 - Random forest regression
    Random_Forest = modeler.regressive_model(train, test, RandomForestRegressor(n_estimators = 10),'RandomForest')

    # Model 3 - XGBoost
    XGBoost = modeler.regressive_model(train, test, XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror'),'XGBoost')

    # Model 4 - Long Short - Term Memory
    LSTM = modeler.lstm_model(train, test)

    # Use Moving average to predict features
    MovingAverage = modeler.MA('Net_Income')

    #Save output to json file
    modeler.to_json(Linear_Regression)
    modeler.to_json(Random_Forest)
    modeler.to_json(XGBoost)
    modeler.to_json(LSTM)
    modeler.to_json(MovingAverage)
