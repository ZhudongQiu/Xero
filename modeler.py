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

def load_data(file_name):
    return pd.read_csv(file_name)

def tts(data):
    data = data.drop(['Net_Income', 'Date'], axis=1)
    train, test = data[0: 22].values, data[-15:].values

    return train, test

def scale_data(train_set, test_set):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_train, y_train = train_set_scaled[:,1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:,1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler

def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):
    y_pred = y_pred.reshape(y_pred.shape[0],1, 1)
    
    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        
    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]], axis=1))
    
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    
    return pred_test_set_inverted

def predict_df(unscaled_predictions, original_df):
    result_list = []
    NI_dates = list(original_df[-16:].Date)
    act_NI = list(original_df[-22:].Net_Income)

    for index in range(0, len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_NI[index])
        result_dict['date'] = NI_dates[index+1]
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)

    return df_result

model_scores = {}

def get_scores(unscaled_df, original_df, model_name):
    rmse = np.sqrt(mean_squared_error(original_df.Net_Income[-15:],unscaled_df.pred_value[-15:]))
    mae = mean_absolute_error(original_df.Net_Income[-15:], unscaled_df.pred_value[-15:])
    r2 = r2_score(original_df.Net_Income[-15:], unscaled_df.pred_value[-15:])
    model_scores[model_name] = [rmse, mae]

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae/100}%")
    print(f"R2 Score: {r2}")
    
def plot_results(results, original_df, model_name):
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(original_df.Date, original_df.Net_Income, data=original_df, ax=ax, label='Original', color='mediumblue')
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax, label='Predicted', color='red')
    ax.set(xlabel='Date', ylabel='Net Income', title=f"{model_name} Net Income Forecasting Prediction")
    ax.legend()
    sns.despine

    plt.savefig(f'{model_name}_forecast.png')

def to_json(x):
    jsonData = json.dumps(x, indent=1)
    fileObject = open('Outcome.json', 'a+')
    fileObject.write(jsonData)
    fileObject.close()

def regressive_model(train_data, test_data, model, model_name):
    
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)
    
    original_df = pd.read_csv('Xero_test.csv')
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)
    
    plot_results(unscaled_df, original_df,model_name)
    get_scores(unscaled_df, original_df, model_name)
    
    a = original_df['Date'][10:]
    b = unscaled_df['pred_value']
        
    pred_df = pd.DataFrame(zip(a,b), columns = ['Date', 'prediction'])
    pred_dic = pred_df.set_index('Date').T.to_dict('list')
    pred_outcome = {model_name: pred_dic}
    
    return pred_outcome

def lstm_model(train_data, test_data):
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
    predictions = model.predict(X_test, batch_size=1)
    
    original_df = pd.read_csv('Xero_test.csv')
    unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
    unscaled_df = predict_df(unscaled, original_df)
    
    get_scores(unscaled_df, original_df, 'LSTM')
    plot_results(unscaled_df, original_df, 'LSTM')
    
    a = original_df['Date'][10:]
    b = unscaled_df['pred_value']
        
    pred_df = pd.DataFrame(zip(a,b), columns = ['Date', 'prediction'])
    pred_dic = pred_df.set_index('Date').T.to_dict('list')
    pred_outcome = {'LSTM':pred_dic}
    
    return pred_outcome

def moving_average(data_array, n=3):
    '''
    Calcuate the moving average based on the specific data array.
    :param data_array: the array stored data to be calculated.
    :param n: the number of data in one time
    :return: Generate which contains the result
     '''
    it = iter(data_array)
    
    d = deque(itertools.islice(it, n - 1))
    s = sum(d)
    # In the first round, to avoid getting extra element, so need zero in the head of queue.
    d.appendleft(0)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / float(n)

def MA(name):
    Xero = pd.read_csv('Xero_test.csv')
    Date = Xero['Date'][2:].values.tolist()
    features = Xero[name].values.tolist()
    features_MA = list(moving_average(features))
    
    #Put these lists together
    a = Date
    b = features_MA   
    pred_df = pd.DataFrame(zip(a,b), columns = ['Date', 'prediction'])
    pred_dic = pred_df.set_index('Date').T.to_dict('list')
    pred_outcome = {'Moving Average':pred_dic}
    
    return pred_outcome

def main():
    """Call function to build models"""
    model_df = pd.read_csv("Xero_features.csv")
    train, test = tts(model_df)

    # Model 1 - Linear regression
    Linear_Regression = regressive_model(train, test, LinearRegression(), 'LinearRegression')

    # Model 2 - Random forest regression
    Random_Forest = regressive_model(train, test, RandomForestRegressor(n_estimators = 10),'RandomForest')

    # Model 3 - XGBoost
    XGBoost = regressive_model(train, test, XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror'),'XGBoost')

    # Model 4 - Long Short - Term Memory
    LSTM = lstm_model(train, test)

    # Use Moving average to predict features
    MovingAverage = MA('Net_Income')

    #Save output to json file
    to_json(Linear_Regression)
    to_json(Random_Forest)
    to_json(XGBoost)
    to_json(LSTM)
    to_json(MovingAverage)

main()
