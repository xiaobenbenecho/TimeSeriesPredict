# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:41:06 2019

@author: yzhang
"""


#def regime_switch_model(df1, inclusion = True):
#    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
#    from math import ceil
#    import warnings
#    warnings.filterwarnings("ignore")
#    #from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
#    import numpy as np
#    
#   
#    
#    ####Split the time series dataset into training and testing################  
#    ####Split the time series dataset into training and testing################  
#    ts_train = df1.iloc[0: ceil(len(df1)*0.9), :]
#    ts_test = df1.iloc[ceil(len(df1)*0.9):, :]
#        
#    mdl = MarkovAutoregression(endog =  ts_train, k_regimes = 4, order = 2, trend = 'nc' )
#    res = mdl.fit()
#
#    print(res.summary())
#    forecast = res.predict()

def state_space_SARIMAX(df1):
    from statsmodels.tsa.arima_model import ARIMA
    import statsmodels.tsa.statespace.sarimax as sm
    from math import ceil
    import warnings
    warnings.filterwarnings("ignore")
    #from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    import numpy as np
    import pandas as pd
    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    
    ####Split the time series dataset into training and testing################  
    df_train = df1.iloc[0: ceil(len(df1)*0.9), :]
    df_test = df1.iloc[ceil(len(df1)*0.9):, :]
    
    
    #find the best ordered ARMA model
    best_hqic = np.inf
    best_order = None
    best_mdl = None
    
    rng = range(5)
    for p in rng:
        for d in rng:
            for q in rng:
                try:
                    tmp_mdl = sm.SARIMAX(df_train, order=(p,d,q)).fit()
                    tmp_hqic = tmp_mdl.hqic
                    if tmp_hqic < best_hqic:
                        best_hqic = tmp_hqic
                        best_order = (p,d,q)
                        best_mdl = tmp_mdl
                except: continue
    #print('hqic: {:6.5f} | order: {}'.format(best_hqic, best_order))
     

#    mdl = sm.SARIMAX(df_train, order = best_order)
#
#    print('BEFORE sarimax MODEL') 
#    res = mdl.fit()
#    print('Aftert sarimax model')
    res = best_mdl
    
    lastdate = str(df_test.index[-1])
        
    #predict =  best_mdl.predict(dynamic = df_test.index[0].to_pydatetime(), end = df_test.index[-1].to_pydatetime())
    predict = res.predict(end = df1.index.get_loc(pd.to_datetime(lastdate)))
    predict = predict[-len(df_test):]

    
    ########Compute the prediction error#############
    pe = prediction_error(df_test.units, predict, original_df = df1, smooth_type = 'normal')
   
    ########Conpute the prediction interva##########
    predict_ci = forecast_pred_int(predict, pe, alpha = 0.05)
    
    #######Assess the goodness of prediction interval########################
    acc_pi, avg_diff_pi = goodness_prediction_interval(df_test, predict_ci)
    
    return pe, acc_pi, avg_diff_pi


#def state_space_SARIMAX(original_df, smoothed_df, smooth_type, inclusion, stationarity):
#    from statsmodels.tsa.arima_model import ARIMA
#    import statsmodels.tsa.statespace.sarimax as sm
#    from math import ceil
#    import numpy as np
#    import pandas as pd
#    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
#    
#    if smooth_type == 'normal':
#        ts = original_df
#    else:
#        ts = smoothed_df
#       
#    if (stationarity == True):
#        ###Split the time series dataset into training and testing################
#        ts_train = ts[0: ceil(len(ts)*0.9)]
#        ts_test = ts[ceil(len(ts)*0.9):]
#        
#        #find the best ordered ARMA model
#        best_hqic = np.inf
#        best_order = None
#        best_mdl = None
#        
#        rng = range(5)
#        for p in rng:
#            for d in rng:
#                for q in rng:
#                    try:
#                        tmp_mdl = ARIMA(ts_train.values, order=(p,d,q)).fit(method = 'mle', trend = 'nc')
#                        tmp_hqic = tmp_mdl.hqic
#                        if tmp_hqic < best_hqic:
#                            best_hqic = tmp_hqic
#                            best_order = (p,d,q)
#                    except: continue
#        #print('hqic: {:6.5f} | order: {}'.format(best_hqic, best_order))
#        
#        best_mdl = sm.SARIMAX(ts_train, order = best_order)
#        res = best_mdl.fit()
#        
#        #.plot_redict function has problem.
#        firstdate = str(ts_test.index[0])
#        lastdate = str(ts_test.index[-1])
#        #ts_predict =  best_mdl.predict(start = ts_test.index[0].to_pydatetime(), end = ts_test.index[-1].to_pydatetime())
#        #ts_predict = best_mdl.predict(start = ts.index.get_loc(pd.to_datetime(firstdate)), end = ts.index.get_loc(pd.to_datetime(lastdate))) 
#        
#        #predict =  best_mdl.predict(dynamic = df_test.index[0].to_pydatetime(), end = df_test.index[-1].to_pydatetime())
#        predict = res.predict(end = smoothed_df.index.get_loc(pd.to_datetime(lastdate)))
#        predict = predict[-len(ts_test):]
#        ########Compute the prediction error#############
#        pe = prediction_error(ts_test.units, predict, original_df = original_df, smooth_type = 'normal')
#   
#        ########Conpute the prediction interva##########
#        predict_ci = forecast_pred_int(predict, pe, alpha = 0.05)
#    
#        #######Assess the goodness of prediction interval########################
#        acc_pi, avg_diff_pi = goodness_prediction_interval(ts_test, predict_ci)
#        
#    else:
#        #####remove trend and seasonality from the time series.#################
#        from stldecompose import decompose, forecast
#        from stldecompose.forecast_funcs import (naive, drift, mean, seasonal_naive)
#        
#        #########################If the length of the ts is shorter than 130#####
#        ########################This is weekly data#############################
#        if len(ts) < 130:
#            stl = decompose(ts, period = 52)
#        else:
#            if (inclusion == False):
#                stl = decompose(ts, period = 251)
#            else:
#                stl = decompose(ts, period = 365)
#        
#        ######Fit ARMA on the Residual##############
#        ts_train = stl.resid[0: ceil(len(stl.resid)*0.9)]
#        ts_test = stl.resid[ceil(len(stl.resid)*0.9):]
#        
#        best_hqic = np.inf
#        best_order = None
#        best_mdl = None
#        
#        rng = range(5)
#        for p in rng:
#            for d in rng:
#                for q in rng:
#                    try:
#                        tmp_mdl = ARIMA(ts_train.values, order=(p,d,q)).fit(method = 'mle', trend = 'nc')
#                        tmp_hqic = tmp_mdl.hqic
#                        if tmp_hqic < best_hqic:
#                            best_hqic = tmp_hqic
#                            best_order = (p,d,q)
#                    except: continue
#        #print('hqic: {:6.5f} | order: {}'.format(best_hqic, best_order))
#        
#        best_mdl = sm.SARIMAX(ts_train, order = best_order)
#        res = best_mdl.fit()
#        
#        #######Prediction#################
#        #firstdate = str(ts_test.index[0])
#        lastdate = str(ts_test.index[-1])
#        
#        #ts_predict =  best_mdl.predict(start = ts_test.index[0].to_pydatetime(), end = ts_test.index[-1].to_pydatetime())
#        #ts_predict = best_mdl.predict(start = ts.index.get_loc(pd.to_datetime(firstdate)), end = ts.index.get_loc(pd.to_datetime(lastdate)))
#        predict = res.predict(end = smoothed_df.index.get_loc(pd.to_datetime(lastdate)))
#        predict = predict[-len(ts_test):]
#        #######Add back the trend and seasonality ########
#        predict = stl.seasonal.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()] + stl.trend.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()] + pd.Series(index = ts_test.index, data = predict)
# 
#        ########Compute the prediction error#############
#        pe = prediction_error(ts_test.units, predict, original_df = original_df, smooth_type = 'normal')
#   
#        ########Conpute the prediction interva##########
#        predict_ci = forecast_pred_int(predict, pe, alpha = 0.05)
#        difference = stl.seasonal.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()] + stl.trend.units.loc[ts_test.index[0].to_pydatetime():ts_test.index[-1].to_pydatetime()]
#        def f(a): return (a + difference)
#        predict_ci = np.apply_along_axis(f, 0, predict_ci)
#        
#        #######Assess the goodness of prediction interval########################
#        acc_pi, avg_diff_pi = goodness_prediction_interval(ts_test, predict_ci)
#        
##    ############Plot the prediction and prediction intervals###################
##    from func_visualisation import plot_prediction
##    plot_prediction(df, prediction, prediction_interval)
##    
#    
#    return pe, acc_pi, avg_diff_pi
#    
#



def state_space_UC(df1):
    import statsmodels.api as sm
    from math import ceil
    import warnings
    warnings.filterwarnings("ignore")
    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    import numpy as np
    import pandas as pd
    
   
    
    ####Split the time series dataset into training and testing################  
    df_train = df1.iloc[0: ceil(len(df1)*0.9), :]
    df_test = df1.iloc[ceil(len(df1)*0.9):, :]
    
    # Fit a local level model
    mdl = sm.tsa.UnobservedComponents(df_train, 'local level', stochastic_trend = True, stochastic_cycle = True, irregular= True)


    res = mdl.fit()
    
    #firstdate = str(df_test.index[0])
    lastdate = str(df_test.index[-1])
        
    #ts_predict =  best_mdl.predict(start = ts_test.index[0].to_pydatetime(), end = ts_test.index[-1].to_pydatetime())
    predict = res.predict(end = df1.index.get_loc(pd.to_datetime(lastdate)))
    predict = predict[-len(df_test):]
    
   ########Compute the prediction error#############
    pe = prediction_error(df_test.units, predict, original_df = df1, smooth_type = 'normal')
   
    ########Conpute the prediction interva##########
    predict_ci = forecast_pred_int(predict, pe, alpha = 0.05)
    
    #######Assess the goodness of prediction interval########################
    acc_pi, avg_diff_pi = goodness_prediction_interval(df_test, predict_ci)
    
    return pe, acc_pi, avg_diff_pi