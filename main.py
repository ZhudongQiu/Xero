# main.py

import mylib

from mylib.dataextraction import *
from mylib.config import *
from mylib.datahandler import *
from mylib.modeler import *


#Dataextraction
data = import_json('xero_json.json')

Date = get_date(data)
Net_Income = get_net_income(data)
Total_Revenue = get_total_revenue(data)
Total_amount_of_Invoice = get_total_amount_of_invoice(data)
Total_amount_of_Bill = get_total_amount_of_bill(data)
Total_Cost_of_Sales = get_total_cost_of_sales(data)
Gross_Profit = get_gross_profit(data)
Total_Operating_Expenses = get_total_operating_expenses(data)
Total_Current_Assets = get_total_current_assets(data)
Total_Assets = get_total_assets(data)
Total_Current_Liabilities = get_total_current_liabilities(data)
Total_Liabilities = get_total_liabilities(data)
Total_Equity = get_total_equity(data)
Total_Liabilities_and_Equity = get_total_liabilities_and_equity(data)

dic = {"Date" : Date,
       "Net_Income" : Net_Income,
       "Total_Revenue" : Total_Revenue,
       "Total_amount_of_Invoice" : Total_amount_of_Invoice,
       "Total_amount_of_Bill" : Total_amount_of_Bill,
       "Total_Cost_of_Sales" : Total_Cost_of_Sales,
       "Gross_Profit" : Gross_Profit,
       "Total_Operating_Expenses" : Total_Operating_Expenses,
       "Total_Current_Assets" : Total_Current_Assets,
       "Total_Current_Liabilities" : Total_Current_Liabilities,
       "Total_Assets" : Total_Assets,
       "Total_Liabilities" : Total_Liabilities,
       "Total_Equity" : Total_Equity,
       "Total_Liabilities_and_Equity" : Total_Liabilities_and_Equity}
               
pd.DataFrame(dic).iloc[::-1].to_csv("Data.csv", index=False)

# Prepare data to build model
info()
    
corr()
    
# Build Model
model_df = pd.read_csv("Features.csv")
train, test = tts(model_df)

# Model 1 - Linear regression
Linear_Regression = regressive_model(train, test, LinearRegression(), 'LinearRegression')

# Model 2 - Random forest regression
Random_Forest = regressive_model(train, test, RandomForestRegressor(n_estimators=Config['RandomForestRegressor_setting']['n_estimators']),'RandomForest')

# Model 3 - XGBoost
XGBoost = regressive_model(train, test, XGBRegressor(n_estimators=Config['XGBRegressor_setting']['n_estimators'],
                                                     learning_rate=Config['XGBRegressor_setting']['learning_rate'],
                                                     objective='reg:squarederror'),'XGBoost')

# Model 4 - Long Short - Term Memory
LSTM = lstm_model(train, test)

# Save output to json file
to_json(Linear_Regression)
to_json(Random_Forest)
to_json(XGBoost)
to_json(LSTM)

# Use Moving average to forecast feature
get_months_feature(mylib.config.Config["Number_months_forecast"])
    
# Import model to predict
test = pd.read_csv('forecast_feature.csv')
model = joblib.load('LinearRegression.pkl')
result = model.predict(test)[3:]

# Save prediction to Json File
month =  get_date_num(mylib.config.Config["Number_months_forecast"])

pred_df = pd.DataFrame(zip(month,result), columns = ['Date', 'prediction'])
pred_dic = pred_df.set_index("Date").T.to_dict('list')
pred_outcome = {"LinearRegression" : pred_dic}

jsonData = json.dumps(pred_outcome, indent=1)
fileObject = open("future_prediction.json", 'a+')
fileObject.write(jsonData)
fileObject.close()
