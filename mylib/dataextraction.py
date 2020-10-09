# dataextraction

"""Get data from json file and input to csv."""
import json

def import_json(name):
    return json.load(open(name,'rb'))

def get_total_current_assets(data):
    total_current_assets = []
    Get_Total_Current_Assets = 0

    i = 0
    while i < 3:
        Get_Total_Current_Assets = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][3]['Rows'][2]['Cells'][1]['Value']
        total_current_assets.append(Get_Total_Current_Assets)
        i = i + 1
    
    while i <= 14:
        Get_Total_Current_Assets = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][3]['Rows'][1]['Cells'][1]['Value']
        total_current_assets.append(Get_Total_Current_Assets)
        i = i + 1

    while len(total_current_assets) < len(data['GetBalanceSheet']):
        total_current_assets.append(0)

    return total_current_assets

def get_total_cost_of_sales(data):
    total_cost_of_sales = []
    Get_Total_Cost_of_Sales = 0

    i = 0
    while i < 3:
        Get_Total_Cost_of_Sales = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][2]['Rows'][1]['Cells'][1]['Value']
        total_cost_of_sales.append(Get_Total_Cost_of_Sales)
        i = i + 1

    while len(total_cost_of_sales) < len(data['GetProfitAndLoss']):
        total_cost_of_sales.append(0)

    return total_cost_of_sales

def get_gross_profit(data):
    gross_profit = []
    Get_Gross_Profit = 0

    i = 0
    while i < 3:
        Get_Gross_Profit = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][3]['Rows'][0]['Cells'][1]['Value']
        gross_profit.append(Get_Gross_Profit)
        i = i + 1
    while i < len(data['GetProfitAndLoss']):
        Get_Gross_Profit = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][2]['Rows'][0]['Cells'][1]['Value']
        gross_profit.append(Get_Gross_Profit)
        i = i + 1
    return gross_profit

def get_total_operating_expenses(data):
    total_operating_expenses = []
    Get_Total_Operating_Expenses = 0

    i = 0
    while i == 0:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][4]['Rows'][0]['Cells'][1]['Value'])
        i = i + 1
    while i == 1:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][4]['Rows'][15]['Cells'][1]['Value'])
        i = i + 1
    while i == 2:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][4]['Rows'][12]['Cells'][1]['Value'])
        i = i + 1
    while i == 3:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][3]['Rows'][10]['Cells'][1]['Value'])
        i = i + 1
    while i == 4:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][3]['Rows'][6]['Cells'][1]['Value'])
        i = i + 1
    while i == 5:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][3]['Rows'][1]['Cells'][1]['Value'])
        i = i + 1
    while i == 6:
        total_operating_expenses.append(data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][3]['Rows'][2]['Cells'][1]['Value'])
        i = i + 1
    while i < 15:
        Get_Total_Operating_Expenses = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][3]['Rows'][1]['Cells'][1]['Value']
        total_operating_expenses.append(Get_Total_Operating_Expenses)
        i = i + 1
        
    while len(total_operating_expenses) < len(data['GetProfitAndLoss']):
        total_operating_expenses.append(0)
        
    return total_operating_expenses

def get_total_assets(data):
    total_assets = []
    Get_Total_Assets = 0

    i = 0
    while i < 7:
        Get_Total_Assets = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][5]['Rows'][0]['Cells'][1]['Value']
        total_assets.append(Get_Total_Assets)
        i = i + 1
    while i < 15:
        Get_Total_Assets = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][4]['Rows'][0]['Cells'][1]['Value']
        total_assets.append(Get_Total_Assets)
        i = i + 1
    while i == 15:
        Get_Total_Assets = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][3]['Rows'][0]['Cells'][1]['Value']
        total_assets.append(Get_Total_Assets)
        i = i + 1
    while len(total_assets) < len(data['GetBalanceSheet']):
        total_assets.append(0)

    return total_assets

def get_total_current_liabilities(data):
    total_current_liabilities = []
    Get_Total_Current_Liabilities = 0

    i = 0
    while i < 7:
        Get_Total_Current_Liabilities = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][8]['Rows'][3]['Cells'][1]['Value']
        total_current_liabilities.append(Get_Total_Current_Liabilities)
        i = i + 1
    while i < 15:
        Get_Total_Current_Liabilities = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][7]['Rows'][3]['Cells'][1]['Value']
        total_current_liabilities.append(Get_Total_Current_Liabilities)
        i = i + 1
    while i == 15:
        total_current_liabilities.append(data['GetBalanceSheet'][i]['Reports'][0]['Rows'][6]['Rows'][2]['Cells'][1]['Value'])
        i = i + 1
    while i < len(data['GetBalanceSheet']):
        Get_Total_Current_Liabilities = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][3]['Rows'][1]['Cells'][1]['Value']
        total_current_liabilities.append(Get_Total_Current_Liabilities)
        i = i + 1

    return total_current_liabilities

def get_total_liabilities(data):
    total_liabilities = []
    Get_Total_Liabilities = 0

    i = 0
    while i < 7:
        Get_Total_Liabilities = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][9]['Rows'][0]['Cells'][1]['Value']
        total_liabilities.append(Get_Total_Liabilities)
        i = i + 1
    while i < 15:
        Get_Total_Liabilities = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][8]['Rows'][0]['Cells'][1]['Value']
        total_liabilities.append(Get_Total_Liabilities)
        i = i + 1
    while i == 15:
        total_liabilities.append(data['GetBalanceSheet'][i]['Reports'][0]['Rows'][7]['Rows'][0]['Cells'][1]['Value'])
        i = i + 1
    while i < len(data['GetBalanceSheet']):
        Get_Total_Liabilities = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][4]['Rows'][0]['Cells'][1]['Value']
        total_liabilities.append(Get_Total_Liabilities)
        i = i + 1

    return total_liabilities

def get_total_equity(data):
    total_equity = []
    Get_Total_Equity = 0

    i = 0
    while i < 7:
        Get_Total_Equity = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][10]['Rows'][4]['Cells'][1]['Value']
        total_equity.append(Get_Total_Equity)
        i = i + 1
    while i == 7:
        total_equity.append(data['GetBalanceSheet'][i]['Reports'][0]['Rows'][9]['Rows'][4]['Cells'][1]['Value'])
        i = i + 1
    while i < 15:
        Get_Total_Equity = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][9]['Rows'][2]['Cells'][1]['Value']
        total_equity.append(Get_Total_Equity)
        i = i + 1
    while i == 15:
        total_equity.append(data['GetBalanceSheet'][i]['Reports'][0]['Rows'][8]['Rows'][1]['Cells'][1]['Value'])
        i = i + 1
    while i < len(data['GetBalanceSheet']):
        Get_Total_Equity = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][5]['Rows'][1]['Cells'][1]['Value']
        total_equity.append(Get_Total_Equity)
        i = i + 1
    return total_equity

def get_total_liabilities_and_equity(data):
    total_liabilities_and_equity = []
    Get_Total_Liabilities_and_Equity = 0

    i = 0
    while i < 7:
        Get_Total_Liabilities_and_Equity = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][11]['Rows'][0]['Cells'][1]['Value']
        total_liabilities_and_equity.append(Get_Total_Liabilities_and_Equity)
        i = i + 1
    while i < 15:
        Get_Total_Liabilities_and_Equity = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][10]['Rows'][0]['Cells'][1]['Value']
        total_liabilities_and_equity.append(Get_Total_Liabilities_and_Equity)
        i = i + 1
    while i == 15:
        total_liabilities_and_equity.append(data['GetBalanceSheet'][i]['Reports'][0]['Rows'][9]['Rows'][0]['Cells'][1]['Value'])
        i = i + 1
    while i < len(data['GetBalanceSheet']):
        Get_Total_Liabilities_and_Equity = data['GetBalanceSheet'][i]['Reports'][0]['Rows'][6]['Rows'][0]['Cells'][1]['Value']
        total_liabilities_and_equity.append(Get_Total_Liabilities_and_Equity)
        i = i + 1
    return total_liabilities_and_equity

def get_net_income(data):
    net_income = []
    Get_Net_Income = 0

    i = 0
    while i < 3:
        Get_Net_Income = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][7]['Rows'][0]['Cells'][1]['Value']
        net_income.append(Get_Net_Income)
        i = i + 1
    
    while i < len(data['GetProfitAndLoss']):
        Get_Net_Income = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][6]['Rows'][0]['Cells'][1]['Value']
        net_income.append(Get_Net_Income)
        i = i + 1
    return net_income

def get_total_revenue(data):
    total_revenue = []
    Get_Total_Revenue = 0

    i = 0
    while i < 7:
        Get_total_Revenue = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][1]['Rows'][1]['Cells'][1]['Value']
        total_revenue.append(Get_total_Revenue)
        i = i + 1
    while i == 7:
        Get_total_Revenue = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][1]['Rows'][2]['Cells'][1]['Value']
        total_revenue.append(Get_total_Revenue)
        i = i + 1
    while i < 15:
        Get_total_Revenue = data['GetProfitAndLoss'][i]['Reports'][0]['Rows'][1]['Rows'][1]['Cells'][1]['Value']
        total_revenue.append(Get_total_Revenue)
        i = i + 1
    while len(total_revenue) < len(data['GetProfitAndLoss']):
        total_revenue.append(0)
    return total_revenue

def search_invoices(data):
    invoices = []
    single_invoice = []
    Type = 0
    Date = 0
    Amount = 0
    i = 0
    while i < len(data['GetInvoices']['Invoices']):
        Type = data['GetInvoices']["Invoices"][i]['Type']
        Date = data['GetInvoices']["Invoices"][i]['Date'][0:7]
        Amount = data['GetInvoices']["Invoices"][i]['Total']
        single_invoice = [Type, Date, Amount]
        invoices.append(single_invoice)
        i = i + 1
    return invoices

def sum_of_month(date):
    sum = 0
    for i in date:
        sum = sum + i
    return sum

def get_total_amount_of_invoice(data):
    date = 0
    get_date = []
    amount_of_invoice = []
    total_amount_of_invoice = []
    b = {}
    month1,month2,month3,month4,month5,month6,month7,month8,month9,month10,month11,month12,month13,month14,month15 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for one in search_invoices(data):
        for j in one:
            if j == "ACCREC":
                del one[0]
                amount_of_invoice.append(one)
                
    for k in amount_of_invoice:
        date = k[0]
        get_date.append(date)
    get_date.sort(reverse=True)
    month = b.fromkeys(get_date)
    date_new = list(month.keys())
    
    for cal in amount_of_invoice:
        if cal[0] == date_new[0]:
            month1.append(cal[1])
        elif cal[0] == date_new[1]:
            month2.append(cal[1])
        elif cal[0] == date_new[2]:
            month3.append(cal[1])
        elif cal[0] == date_new[3]:
            month4.append(cal[1])
        elif cal[0] == date_new[4]:
            month5.append(cal[1])
        elif cal[0] == date_new[5]:
            month6.append(cal[1])
        elif cal[0] == date_new[6]:
            month7.append(cal[1])
        elif cal[0] == date_new[7]:
            month8.append(cal[1])
        elif cal[0] == date_new[8]:
            month9.append(cal[1])
        elif cal[0] == date_new[9]:
            month10.append(cal[1])
        elif cal[0] == date_new[10]:
            month11.append(cal[1])
        elif cal[0] == date_new[11]:
            month12.append(cal[1])
        elif cal[0] == date_new[12]:
            month13.append(cal[1])
        elif cal[0] == date_new[13]:
            month14.append(cal[1])
        elif cal[0] == date_new[14]:
            month15.append(cal[1])
        
    total_amount_of_invoice = [sum_of_month(month1),sum_of_month(month2),
                               sum_of_month(month3),sum_of_month(month4),
                               sum_of_month(month5),sum_of_month(month6),
                               sum_of_month(month7),sum_of_month(month8),
                               sum_of_month(month9),sum_of_month(month10),
                               sum_of_month(month11),sum_of_month(month12),
                               sum_of_month(month13),sum_of_month(month14),
                               sum_of_month(month15)]
    
    while len(total_amount_of_invoice) < len(data['GetProfitAndLoss']):
        total_amount_of_invoice.append(0)
    
    return total_amount_of_invoice

def get_total_amount_of_bill(data):
    date = 0
    get_date = []
    amount_of_bill = []
    total_amount_of_bill = []
    b = {}
    month1,month2,month3,month4,month5,month6,month7,month8,month9,month10,month11,month12,month13,month14,month15 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for one in search_invoices(data):
        for j in one:
            if j == "ACCPAY":
                del one[0]
                amount_of_bill.append(one)
                
    for k in amount_of_bill:
        date = k[0]
        get_date.append(date)
    get_date.sort(reverse=True)
    month = b.fromkeys(get_date)
    date_new = list(month.keys())
    
    for cal in amount_of_bill:
        if cal[0] == date_new[0]:
            month1.append(cal[1])
        elif cal[0] == date_new[1]:
            month2.append(cal[1])
        elif cal[0] == date_new[2]:
            month3.append(cal[1])
        elif cal[0] == date_new[3]:
            month4.append(cal[1])
        elif cal[0] == date_new[4]:
            month5.append(cal[1])
        elif cal[0] == date_new[5]:
            month6.append(cal[1])
        elif cal[0] == date_new[6]:
            month7.append(cal[1])
        elif cal[0] == date_new[7]:
            month8.append(cal[1])
        elif cal[0] == date_new[8]:
            month9.append(cal[1])
        elif cal[0] == date_new[9]:
            month10.append(cal[1])
        elif cal[0] == date_new[10]:
            month11.append(cal[1])
        elif cal[0] == date_new[11]:
            month12.append(cal[1])
        elif cal[0] == date_new[12]:
            month13.append(cal[1])
        elif cal[0] == date_new[13]:
            month14.append(cal[1])
        elif cal[0] == date_new[14]:
            month15.append(cal[1])
        
    total_amount_of_bill = [sum_of_month(month1),sum_of_month(month2),
                               sum_of_month(month3),sum_of_month(month4),
                               sum_of_month(month5),sum_of_month(month6),
                               sum_of_month(month7),sum_of_month(month8),
                               sum_of_month(month9),sum_of_month(month10),
                               sum_of_month(month11),sum_of_month(month12),
                               sum_of_month(month13),sum_of_month(month14),
                               sum_of_month(month15)]

    while len(total_amount_of_bill) < len(data['GetProfitAndLoss']):
        total_amount_of_bill.append(0)

    return total_amount_of_bill
        
def get_date(data):
    date = []

    i = 0
    while i < len(data['GetBalanceSheet']):
        date.append(data['GetBalanceSheet'][i]["Reports"][0]['Rows'][0]['Cells'][1]['Value'][3:11])
        i = i + 1

    return date
