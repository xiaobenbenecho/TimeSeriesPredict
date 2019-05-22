# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:54:30 2019

@author: yzhang
"""

#ts = pd.read_pickle("C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\ts.pkl")
#inclusion = True
#smooth_type = 'normal'
#agg_level = 'W'

#####input: Data frame without missing/absent values###########################
#####function:realise the single time series analysis##########################
#original_df = df0
#smoothed_df = df1
#stationarity = stationarity_status
def automation_single_ts_arma_analysis(original_df, smoothed_df, smooth_type, inclusion, stationarity):
    from statsmodels.tsa.arima_model import ARIMA
    from math import ceil
    import numpy as np
    import pandas as pd
    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    
    if smooth_type == 'normal':
        ts = original_df
    else:
        ts = smoothed_df
       
    if (stationarity == True):
        ###Split the time series dataset into training and testing################
        ts_train = ts[0: ceil(len(ts)*0.9)]
        ts_test = ts[ceil(len(ts)*0.9):]
        
        #find the best ordered ARMA model
        best_hqic = np.inf
        best_order = None
        best_mdl = None
        
        rng = range(5)
        for p in rng:
            for d in rng:
                for q in rng:
                    try:
                        tmp_mdl = ARIMA(ts_train.values, order=(p,d,q)).fit(method = 'mle', trend = 'nc')
                        tmp_hqic = tmp_mdl.hqic
                        if tmp_hqic < best_hqic:
                            best_hqic = tmp_hqic
                            best_order = (p,d,q)
                            best_mdl = tmp_mdl
                    except: continue
        #print('hqic: {:6.5f} | order: {}'.format(best_hqic, best_order))
        
        
        #.plot_redict function has problem.
        firstdate = str(ts_test.index[0])
        lastdate = str(ts_test.index[-1])
        #ts_predict =  best_mdl.predict(start = ts_test.index[0].to_pydatetime(), end = ts_test.index[-1].to_pydatetime())
        #ts_predict = best_mdl.predict(start = ts.index.get_loc(pd.to_datetime(firstdate)), end = ts.index.get_loc(pd.to_datetime(lastdate))) 
            
        ###calcualte the prediction interval.     
        ts_forecast, std_error, prediction_interval = best_mdl.forecast(len(ts_test))
        
    else:
        #####remove trend and seasonality from the time series.#################
        from stldecompose import decompose, forecast
        from stldecompose.forecast_funcs import (naive, drift, mean, seasonal_naive)
        
        #########################If the length of the ts is shorter than 130#####
        ########################This is weekly data#############################
        if len(ts) < 130:
            stl = decompose(ts, period = 52)
        else:
            if (inclusion == False):
                stl = decompose(ts, period = 251)
            else:
                stl = decompose(ts, period = 365)
        
        ######Fit ARMA on the Residual##############
        ts_train = stl.resid[0: ceil(len(stl.resid)*0.9)]
        ts_test = stl.resid[ceil(len(stl.resid)*0.9):]
        
        best_hqic = np.inf
        best_order = None
        best_mdl = None
        
        rng = range(5)
        for p in rng:
            for d in rng:
                for q in rng:
                    try:
                        tmp_mdl = ARIMA(ts_train.values, order=(p,d,q)).fit(method = 'mle', trend = 'nc')
                        tmp_hqic = tmp_mdl.hqic
                        if tmp_hqic < best_hqic:
                            best_hqic = tmp_hqic
                            best_order = (p,d,q)
                            best_mdl = tmp_mdl
                    except: continue
        #print('hqic: {:6.5f} | order: {}'.format(best_hqic, best_order))
        
 
        #######Prediction#################
        firstdate = str(ts_test.index[0])
        lastdate = str(ts_test.index[-1])
        
        #ts_predict =  best_mdl.predict(start = ts_test.index[0].to_pydatetime(), end = ts_test.index[-1].to_pydatetime())
        ts_predict = best_mdl.predict(start = ts.index.get_loc(pd.to_datetime(firstdate)), end = ts.index.get_loc(pd.to_datetime(lastdate)))
        
        #######Add back the trend and seasonality ########
        ts_predict = stl.seasonal.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()] + stl.trend.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()] + pd.Series(index = ts_test.index, data = ts_predict)
 
        #########Compute the prediction interval
        ts_forecast, std_error, prediction_interval = best_mdl.forecast(len(ts_test))
        difference = stl.seasonal.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()] + stl.trend.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()]
        def f(a): return (a + difference)
        prediction_interval = np.apply_along_axis(f, 0, prediction_interval)
        
    ########Compute the prediction error#############
    pe = prediction_error(ts_test.units, ts_forecast, original_df = original_df, smooth_type = smooth_type)
   
    #######Assess the goodness of prediction interval########################
    acc_pi, avg_diff_pi = goodness_prediction_interval(ts_test, prediction_interval)
    
    
#    ############Plot the prediction and prediction intervals###################
#    from func_visualisation import plot_prediction
#    plot_prediction(df, prediction, prediction_interval)
#    
    
    return best_order, pe, acc_pi, avg_diff_pi
    

