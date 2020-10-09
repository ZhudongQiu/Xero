# datahandler.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
This file is used for data preprocessing and data preparation for forecasting.

'''

def load_data(file_name):
    """Import dataframe."""
    return pd.read_csv(file_name)

def info():
    """To see the dataframe's basic information."""
    Xero = load_data('Data.csv')
    Xero['Date'] = pd.to_datetime(Xero['Date'])
    Xero.info()
    Xero.Net_Income.describe()

    plt.figure(figsize = (10,5))
    print('skew: ', Xero.Net_Income.skew())
    sns.distplot(Xero['Net_Income'])
    # plt.show()
    

def corr():
    """Data preparation for forecasting."""
    Xero = load_data('Data.csv')
    corrMat = Xero.corr()
    mask = np.array(corrMat)
    mask[np.tril_indices_from(mask)] = False
    plt.subplots(figsize=(20,10))
    sns.heatmap(corrMat, mask=mask, vmax=0.8, square=True, annot=True)
    # plt.show()

    print(corrMat['Net_Income'].sort_values(ascending=False))

    dic = corrMat['Net_Income'].to_dict()
    lis = []
    for i in dic.values():
        if abs(i) < 0.4:
            lis.append(i)
    for j in dic.items():
        for k in lis:
            if j[1] == k:
                del Xero[j[0]]
    Xero['Date'] = pd.to_datetime(Xero['Date'])
    Xero.to_csv('Features.csv', index=False)
