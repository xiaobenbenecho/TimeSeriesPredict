# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:02:05 2019

@author: yzhang
"""

#################INPUTS: the original datafram and the time aggregation level 
#################OUTPUTS: the dataframe whose index is the list of all feasible products.
#################FUNCTION: Filter the feasible from all products.##############
def filter_feasible_product(data, cutoff= None, agg_level = 'D', agg_def = ['country', 'vendor', 'product_range'], select = True):
    if cutoff is None:
        if agg_level == 'D':
            cutoff_obs_no, cutoff_sparse_ratio = 400, 0.6
        elif agg_level == 'W':
            cutoff_obs_no, cutoff_sparse_ratio = 56, 0.9
        else:
            pass
    else:
        cutoff_obs_no, cutoff_sparse_ratio = cutoff[0], cutoff[1]
        
        
    import pandas as pd
    
    if agg_level == 'D':
        ############Daily Level prediction#############################################
        ##############filter product that has more than 400 days and contains less than 40% missing/absent values of data#################
        f = {'date':['min', 'max', 'count', pd.Series.nunique], 'units':['sum']}
        data_grouped = data.groupby(agg_def)
        summary_data_grouped = data_grouped.agg(f)
        summary_data_grouped.columns = ['startDate','endDate','noRecords','noDays','units']

        
        summary_data_grouped['period'] = (pd.to_datetime(summary_data_grouped['endDate']) - pd.to_datetime(summary_data_grouped['startDate'])).dt.days
       
        if select == True:
            temp = summary_data_grouped
            temp = temp.sort_values('period', ascending = False)
            temp = temp[temp['period'] >= cutoff_obs_no]
            temp = temp[temp['noDays']/temp['period'] >= cutoff_sparse_ratio]
        else:
            temp = summary_data_grouped
     
    else:#if agg_level = 'W'
############Weekly level prediction############################################
##############filter product that has more than 400 days and contains less than 40% missing/absent values of data#################
        data['date'] = pd.to_datetime(data['date'])
        import datetime as dt
        data['date'] = data['date'] - data['date'].dt.weekday.astype('timedelta64[D]')
        
        data_grouped = data.groupby(agg_def)
        f = {'date':['min', 'max', pd.Series.nunique], 'units':['sum']}
        summary_data_grouped = data_grouped.agg(f)
        summary_data_grouped.columns = ['startDate','endDate', 'noWeeks','units']
        
        from math import ceil
        summary_data_grouped['period'] = ((summary_data_grouped['endDate'] - summary_data_grouped['startDate']).dt.days)/7
        
        if select == True:   
            temp = summary_data_grouped[(summary_data_grouped.noWeeks/summary_data_grouped.period  > cutoff_sparse_ratio)  & (summary_data_grouped.period > cutoff_obs_no)]
        else:
             temp = summary_data_grouped
    return temp, data_grouped, len(summary_data_grouped)

# Stationarity tests
def test_stationarity(timeseries):
    from statsmodels.tsa.stattools import adfuller
    import pandas as pd
    #Perform Dickey-Fuller test:
    #print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    if dfoutput.loc['p-value'] < 0.05:
        return True
    else: return False
################################################################################

def outliers_iqr(ys):
    import numpy as np
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return ys.index[(ys.iloc[:,0] > upper_bound) | (ys.iloc[:,0] < lower_bound)]

################################################################################
#ASsessment of the prediction results on all feasible products
###############################################################################
def prediction_error(true, prediction, original_df, method = 'rmse', smooth_type = 'normal'):
    from sklearn.metrics import mean_squared_error
    from math import sqrt


    if smooth_type == 'bin':
        from func_pre_process import pre_process_ts_bin
        df_smooth, bins, group_names = pre_process_ts_bin(original_df)
        prediction_bin = (pd.cut(prediction, bins, labels = group_names)).astype(float)
        pe = sum(true == prediction_bin)/len(true)
        
        #prediction_bin = (pd.cut(ts_predict, bins, labels = group_names)).astype(float)
        #pe = sum(ts_test.units == prediction_bin)/len(ts_test)
        return pe
    else: #smooth_type == 'normal':
        rmse = sqrt(mean_squared_error(original_df.tail(len(prediction))[original_df.columns[0]], prediction))
        return rmse

def forecast_pred_int(forecast, fcasterr, alpha = 0.05):
    from scipy.stats import norm
    import numpy as np
    const = norm.ppf(1 - alpha / 2.)
    conf_int = np.c_[forecast - const * fcasterr,
                     forecast + const * fcasterr]

    return conf_int

def goodness_prediction_interval(ts_test, prediction_interval):
    import numpy as np
    from statistics import mean
    
    if type(ts_test) != list:
        acc_pi = np.sum((ts_test.units >= prediction_interval[:,0])&(ts_test.units <= prediction_interval[:,1]))/len(ts_test)
    else:
        acc_pi = np.sum((ts_test >= prediction_interval[:,0])&(ts_test <= prediction_interval[:,1]))/len(ts_test)
    avg_diff_pi = mean(prediction_interval[:, 1] - prediction_interval[:, 0])
    return acc_pi, avg_diff_pi
