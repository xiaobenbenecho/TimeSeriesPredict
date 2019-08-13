# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:40:54 2019

@author: yzhang

"""




##############enchmark naive method: y_{T+h} = y_{T}
def _naive(original_df, smoothed_df, smooth_type):
    from math import ceil, sqrt
    import numpy as np
    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    
    if smooth_type == 'normal':
        df = original_df
    else:
        df = smoothed_df
        
    #split to training and test set
    df_train = df[0: ceil(len(df)*0.9)]
    df_test = df[ceil(len(df)*0.9):]
    
    #get the prediction series of the set.
    prediction = df_test.copy()
    mdl = df_train.iloc[-1].units
    prediction.units = mdl
 
    #compute prediction error
    pe = prediction_error(df_test.units, prediction.units, original_df = original_df, smooth_type = smooth_type)
    
    
    ########Calcualte prediction intervals######
    h = np.arange(1, len(df_test) + 1, 1)
    fcasterr = np.std(df_test.units - prediction.units)*np.sqrt(h)
    prediction_interval = forecast_pred_int(mdl, fcasterr, alpha = 0.05)
    

    #######Assess the goodness of prediction interval########################
    acc_pi, avg_diff_pi = goodness_prediction_interval(df_test, prediction_interval)
    
#    ############Plot the prediction and prediction intervals###################
#    from func_visualisation import plot_prediction
#    plot_prediction(df, prediction, prediction_interval)
#    
    return mdl, pe, acc_pi, avg_diff_pi


##############Benchmark average method: y_{T+h} = \bar{y}######################
#df =  ts
def _average(original_df, smoothed_df, smooth_type):
    #split to training and test set
    from math import ceil,sqrt
    import numpy as np
    import pandas as pd 
    from functions import goodness_prediction_interval, forecast_pred_int, prediction_error
    
    if smooth_type == 'normal':
        df = original_df
    else:
        df = smoothed_df
    
    
    #split to training and test set
    df_train = df[0: ceil(len(df)*0.9)]
    df_test = df[ceil(len(df)*0.9):]
    
    
    #calcuale the average of training set.
    mdl = df_train.units.mean()
    prediction = df_test.copy()
    prediction.loc[:,'units'] = mdl
    
    #compute prediction error
    pe = prediction_error(df_test.units, prediction.units, original_df = original_df, smooth_type = smooth_type)
    
    ########Calcualte the prediction intervals######
    fcasterr = [np.std(df_test.units - prediction.units)*sqrt(1+1/len(df_train))]*len(df_test)
    prediction_interval = forecast_pred_int(mdl, pd.Series(fcasterr), alpha = 0.05)
   
    #######Assess the goodness of prediction interval########################
    acc_pi, avg_diff_pi = goodness_prediction_interval(df_test, prediction_interval)
    
#    ############Plot the prediction and prediction intervals###################
#    from func_visualisation import plot_prediction
#    plot_prediction(df, prediction, prediction_interval)
#    
    
    return mdl, pe, acc_pi, avg_diff_pi