# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:50:25 2019

@author: yzhang
"""
#df = df1
#df = copy.deepcopy(df0)
def MTM_model(df):
    #input: the df for analysis:
    #output: the pe, acc_pi, width_pi
    
    import pandas as pd
    import numpy as np
    import copy
    from math import ceil
    

    #############STEPS:
    #convert the df to binned values, with bin name as the average of bin boundaries    
    number_quantile = min(ceil(len(df)/4), 100)
    
    while True :
        bins = np.unique(np.quantile(df.iloc[:, 0], np.arange(0, number_quantile+1)/number_quantile)) 
        group_names = []
        corr_bin_name = []
        bin_units = []
        for i in range(1,len(bins)):
            name = (bins[i-1]+ bins[i])/2
            group_names.append(name)
 
        bins[0] = bins[0] - 0.1 
        
        #the value that maps the observation to a unique bin
        bin_units = (pd.cut(df.units, bins, labels = group_names)).astype(float)
        #the bin number
        corr_bin_name =  (pd.cut(df.units, bins, labels = range(1, len(bins)))).astype(int)
        number_quantile = ceil(number_quantile/2)
        if 1 not in pd.Series(corr_bin_name).value_counts().values:
            break
    
    #check whether bins and group names is unique
    if not pd.Series(bins).is_unique:
        print (bins)
    if not pd.Series(group_names).is_unique:
        print (group_names)
    
    df.units = bin_units
 
    
    #calcuate the MTM
    df_train = df[0: ceil(len(df)*0.9)]
    df_test = df[ceil(len(df)*0.9):]
    
        ###Input: the converted time series, the bins.    
    n = len(group_names)
    
    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(corr_bin_name, corr_bin_name[1:len(df_train)+1]):
        #print (i,j)
        M[i-1][j-1] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    
    
    #do the prediction and calcualte required values.
    last = group_names.index(df_train.iloc[-1, 0])
    prediction = []
    
    for i in range(1, len(df_test) + 1):
        temp = M[last].index(max(M[last]))

        #####calcualte the prediction value.
        last = copy.deepcopy(temp)
        prediction.append(group_names[temp])
   
    #calculate the prediction error
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmse = sqrt(mean_squared_error(df_test, prediction)) 
    
    #####calcualte the prediction interval#######
    from functions import forecast_pred_int, goodness_prediction_interval
    prediction_interval = forecast_pred_int(prediction, rmse, alpha = 0.05)
    acc_pi, width_pi = goodness_prediction_interval(df_test, prediction_interval)
      
    return rmse, acc_pi, width_pi



#potential risks: the sequence might report error due to the input is dataframe.
#seq = df_bin.T.squeeze()
def MTM_window(seq, n = 20):
    from itertools import islice
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def MTM_binning(df):
    
    import pandas as pd
    import numpy as np
    from math import ceil
    

    #############STEPS:
    #convert the df to binned values, with bin name as the average of bin boundaries    
    number_quantile = min(ceil(len(df)/4), 100)
    
    while True :
        bins = np.unique(np.quantile(df.iloc[:, 0], np.arange(0, number_quantile+1)/number_quantile)) 
        group_names = []
        corr_bin_name = []
        bin_units = []
        for i in range(1,len(bins)):
            name = (bins[i-1]+ bins[i])/2
            group_names.append(name)
 
        bins[0] = bins[0] - 0.1 
        
        #the value that maps the observation to a unique bin
        bin_units = (pd.cut(df.units, bins, labels = group_names)).astype(float)
        #the bin number
        corr_bin_name =  (pd.cut(df.units, bins, labels = range(1, len(bins)))).astype(int)
        number_quantile = ceil(number_quantile/2)
        if 1 not in pd.Series(corr_bin_name).value_counts().values:
            break
    
    #check whether bins and group names is unique
    if not pd.Series(bins).is_unique:
        print (bins)
    if not pd.Series(group_names).is_unique:
        print (group_names)
    
    df.units = bin_units
    
    return df, corr_bin_name, group_names
 
def MTM_matrix(df_window, corr_bin_name, group_names):
    ###Input: the converted time series, the bins.    
    n = len(group_names)
    
    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(corr_bin_name, corr_bin_name[1:len(df_window)+1]):
        #print (i,j)
        M[i-1][j-1] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    
    return M

def MTM_forecasts(df, time_horizen):
    #do the prediction and calcualte required values.
    last = group_names.index(df_train.iloc[-1, 0])
    prediction = []
    
    for i in range(1, len(df_test) + 1):
        temp = M[last].index(max(M[last]))

        #####calcualte the prediction value.
        last = copy.deepcopy(temp)
        prediction.append(group_names[temp])
   
    #calculate the prediction error
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rmse = sqrt(mean_squared_error(df_test, prediction)) 
    return rmse, prediction[0]
  
    
    

#df = df1.copy()
    
def MTM_compute_pe_window(df_bin, corr_bin_name, group_names, window_size): 
#    #for each window:
#        #1. calculate the MTM.(if Row == 0, then do random jump)
#        #2. calcualte the forecasting error
#        #3. return forecsting error and the one-step ahead forecasting
#    
#    #calcualte the average pe, calcualte the prediction interval according to 
#    #one-step ahead forecasting.
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from functions import forecast_pred_int, goodness_prediction_interval
    import random
    random.seed(4)
    import numpy as np
    
    ts_test = []
    ts_forecast = []
    
    
    for df_window in MTM_window(df_bin.T.squeeze(), window_size + 1):
        #print (df_window)
        M = MTM_matrix(df_window[: -2], corr_bin_name, group_names)
        #df_window = list(df_window)
        
        last = group_names.index(df_window[-2])
        ts_test.append(df_window[-2])
        
        if not np.any(M[last]):##if M[last] is an array of 0
            one_step_forecast = random.choice(range(0, len(M)))
        else:
            one_step_forecast = M[last].index(max(M[last]))
        
        one_step_forecast = group_names[one_step_forecast]
        ts_forecast.append(one_step_forecast)
        
   
    pe = sqrt(mean_squared_error(ts_test, ts_forecast))
     
    pi = forecast_pred_int(ts_forecast, pe, alpha = 0.05)
    
    pi_cr, pi_width = goodness_prediction_interval(ts_test, pi)
    
    return pe, pi_cr, pi_width
  


def MTM_slides_window(df, inclusion = True):
     #categorise the observations into bins and assign obervations with bin indicator
#    
#    #determine window_size and time_horizen
    import numpy as np
    
    df_bin, corr_bin_name, group_names = MTM_binning(df)

    
    #choose the optimal window size
    optimal_window_width = None
    optimal_pe = np.inf
    
    #for daily data
    if len(df_bin) > 400:
        if inclusion == True:
            window_size_list = [100, 180, 365]
        else:
            window_size_list = [65, 130, 260]
    #for weekly data 
    else:
        window_size_list = [12, 26, 52]
    

    for window_size in window_size_list:
        #perform analysis for each window size 
        pe, pi_cr, pi_width = MTM_compute_pe_window(df_bin, corr_bin_name, group_names, window_size)
        
        if pe < optimal_pe:
            optimal_pe = pe
            optimal_pi_cr =  pi_cr
            optimal_pi_width = pi_width
            optimal_window_width = window_size
            
    return  optimal_window_width, optimal_pe, optimal_pi_cr, optimal_pi_width



















  