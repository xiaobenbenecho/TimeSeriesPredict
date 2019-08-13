# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:38:54 2019

@author: yzhang
"""


#############################EDA on the selected products###############################
#1. Aggregate products, Product ID: country, vendor_id, product_id. on daily or weekly level.
#2. Filtering feasible product. (percentage within all feasible products, percentage of the feasible product within the total revenue, percentage of the feasible product within the total sales units, and also the percentaegs considered in each country and vendor).
#3. Analysis on the filtered data. (missing values of each columns, non-existent values by weekdays in each country, sale volume by weekdays in each country, )
#4. After the filtering, aggregation on a daily or weekly basis.

###############Distribution of the products according to noDays#################
def aggregation(data, agg_def):
    import pandas as pd
    f = {'date':['min', 'max', 'count', pd.Series.nunique], 'units':['sum'], 'revenue':['sum']}
    data_grouped = data.groupby(agg_def)
    summary_data_grouped = data_grouped.agg(f)
    summary_data_grouped.columns = ['startDate','endDate','noRecords','noDays','units', 'revenue']
    return summary_data_grouped



###############Check whether the feasible products are top sellers by calcualting the ratio of units
def top_seller_by_country_vendor(data, summary_data, category = ['country', 'vendor']):
    f = {'units':['sum']}
    summary1 = data.groupby(category).agg(f)
    summary2 = summary_data.reset_index()
    summary2 = summary2.groupby(category).agg(f)
    summary = summary2.merge(summary1, left_on = category, right_on = category)
    summary.columns = ['units_feasible_products', 'total_units']
    summary['ratio'] = summary['units_feasible_products']/summary['total_units']
    return summary



###############Sale volume by weekdays or each month in each country ##########
def units_on_weekday_or_month(data, agg_type = 'month', agg_def = ['country','vendor','product']):
    import pandas as pd
    import numpy as np
    
    aggregation = agg_def
    aggregation.insert(0, 'date')

    
    f = {'units':['sum'], 'revenue':['sum']}
    summary = data.groupby(aggregation).agg(f)
    summary1 = summary.reset_index()
    
    summary1['date'] = pd.to_datetime(summary1['date'])
    
    if agg_type == 'month':
        summary1['month'] = summary1.date.dt.strftime("%m")
    elif agg_type == 'weekday':
        summary1['weekday'] = summary1.date.dt.strftime("%A")
    
    summary1.columns = ['date','country','vendor','product', 'units', 'revenue', agg_type]
    return summary1


def nonexistent_values_weekday(data, agg_def = ['country','vendor','product']):
    import pandas as pd
    import numpy as np
    
    
    data['date'] = pd.to_datetime(data['date'])
    data['weekday'] = data.date.dt.strftime("%A")
    

    aggregation = agg_def
    aggregation.append('weekday')
    
    f = {'units':['sum']}
    summary = data.groupby(aggregation).agg(f)
    summary.columns = ['units']
    
    nonexistent_values = summary.unstack(level = -1)
    number_0sales = nonexistent_values['units'].isna().sum()
    
    return number_0sales
    
    
    








