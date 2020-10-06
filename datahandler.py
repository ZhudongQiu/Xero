# datahandler.py

'''
This file is used for data preprocessing and data preparation for forecasting.

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_name):
    """Import dataframe."""
    return pd.read_csv(file_name)

def info():
    """To see the dataframe's basic information."""
    Xero = load_data('Xero_train.csv')
    Xero.info()
    Xero.Net_Income.describe()

    plt.figure(figsize = (10,5))
    print('skew: ', Xero.Net_Income.skew())
    sns.distplot(Xero['Net_Income'])
    # plt.show()
    

def corr():
    """Data preparation for forecasting."""
    Xero = load_data('Xero_train.csv')
    corrMat = Xero.corr()
    mask = np.array(corrMat)
    mask[np.tril_indices_from(mask)] = False
    plt.subplots(figsize=(20,10))
    sns.heatmap(corrMat, mask=mask, vmax=0.8, square=True, annot=True)
    # plt.show()

    print(corrMat['Net_Income'].sort_values(ascending=False))

    del Xero['Total_Cost_of_Sales']
    del Xero['Total_Liabilities_and_Equity']
    del Xero['Total_Current_Liabilities']
    del Xero['Total_Current_Assets']

    Xero.to_csv('Xero_features.csv', index=False)
