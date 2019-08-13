# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:52:16 2019

@author: yzhang
"""

#############This is a file for Univariate Model forecasting###################
#############input: single column dataframe, index is the date/time###########
#ts =  df
def automation_prophet_model(original_df, smoothed_df, smooth_type,  changepoint_number = None, changepoint_scale= None, inclusion = True):
    from fbprophet import Prophet 
    from math import ceil
    import warnings
    warnings.filterwarnings("ignore")
    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    import numpy as np
      
    if smooth_type == 'normal':
        ts = original_df
    else:
        ts = smoothed_df
   
    
    ####Split the time series dataset into training and testing################
    ts = ts.reset_index()
    ts.columns = ['ds','y']
    
    ts_train = ts.iloc[0: ceil(len(ts)*0.9), :]
    ts_test = ts.iloc[ceil(len(ts)*0.9): , :]
    
    ######fit the prophet model################################################
    if (changepoint_number is not None)&(changepoint_scale is not None):
        mdl = Prophet(daily_seasonality = False, interval_width = 0.95, changepoint_prior_scale = changepoint_scale,  n_changepoints = changepoint_number)
    elif changepoint_number is not None:
        mdl = Prophet(daily_seasonality = False, interval_width = 0.95, n_changepoints = changepoint_number)  
    elif changepoint_scale is not None:
        mdl = Prophet(daily_seasonality = False, interval_width = 0.95, changepoint_prior_scale = changepoint_scale) 
    else:
        mdl = Prophet(daily_seasonality = False, interval_width = 0.95)
        
        
    if (inclusion == False):
        mdl.add_seasonality(name='weekly', period = 5, fourier_order = 12)
        mdl.add_seasonality(name='yearly', period = 260.89, fourier_order = 12)


    mdl.fit(ts_train)
    
    future = mdl.make_future_dataframe(periods = len(ts_test))
    forecast = mdl.predict(future)
    
    ts_predict = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(len(ts_test))

    #compute the prediction error##########################
    pe = prediction_error(ts_test['y'], ts_predict['yhat'], original_df = original_df, smooth_type = smooth_type)
    
    #######Assess the goodness of prediction interval########################
    import numpy as np
    from statistics import mean
    acc_pi = np.sum((ts_test['y'] >= ts_predict['yhat_lower'])&(ts_test['y'] <= ts_predict['yhat_upper']))/len(ts_test)
    avg_diff_pi = mean(ts_predict['yhat_upper'] - ts_predict['yhat_lower'])
    
    
    
    return mdl, pe, acc_pi, avg_diff_pi
  