# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:38:36 2019

@author: yzhang
"""

""

""





####################Inputs: orignal dataframe, smoothed dataframe.
####################smooth_type: whether the data has been smoothed################
####################inclusion: whether to include or exclusde weekends data####
####################stationarity: the stationariyt status of the dataframe#####
def automation_model_iteration(original_df, smoothed_df, smooth_type, inclusion, stationarity):
    from func_ARIMA import automation_single_ts_arma_analysis
    from func_prophet_model import automation_prophet_model
    from func_benchmarks import _naive, _average
    from func_MTM import MTM_model, MTM_slides_window
    

    ##for each model
    ##RETURN: The model itself, the prediction, the prediction intervals 
    arma_mdl, pe_arma, acc_pi_arma, avg_diff_pi_arma = automation_single_ts_arma_analysis(original_df, smoothed_df, smooth_type, inclusion, stationarity)
    
    #Prophet model result
    prophet_mdl, pe_prophet, acc_pi_prophet, avg_diff_pi_prophet = automation_prophet_model(original_df, smoothed_df, smooth_type, inclusion)
    
    #MTM model results
    pe_MTM, acc_pi_MTM, avg_diff_pi_MTM = MTM_model(smoothed_df)
    
    
    #MTM with rolling window results
    window_size, pe, acc_pi, avg_diff_pi = MTM_slides_window(smoothed_df, inclusion)
    
    #benchmark model: naive
    naive_mdl, pe_naive, acc_pi_naive, avg_diff_pi_naive = _naive(original_df, smoothed_df, smooth_type)
    
    #benchmark model: average
    average_mdl, pe_average, acc_pi_average, avg_diff_pi_average = _average(original_df, smoothed_df, smooth_type)
    
    
    
    return (arma_mdl, pe_arma, acc_pi_arma, avg_diff_pi_arma,
            pe_prophet, acc_pi_prophet, avg_diff_pi_prophet,
            pe_MTM, acc_pi_MTM, avg_diff_pi_MTM,
            window_size, pe, acc_pi, avg_diff_pi,
            pe_naive, acc_pi_naive, avg_diff_pi_naive,
            pe_average, acc_pi_average, avg_diff_pi_average)


def automation(index_list, data_grouped, agg_level = 'D', smooth_type = 'normal'):
    import warnings
    warnings.filterwarnings("ignore")
    from functions import test_stationarity
    from func_pre_process import pre_process_ts_bin, pre_process_spikes, pre_process_fill_absent_values
    import copy
    import pandas as pd
    #data_filter = data_bigger500
    #build an dataframe to contain the returned value with index to the indication of the product ID.
    df_return_inclusive = pd.DataFrame()
    df_return_exclusive = pd.DataFrame()
    df_outliers = pd.DataFrame()


    #do the iteration for every group in the filtered dataset.
    for index in index_list:
        index = tuple(index)
        data_example = data_grouped.get_group(index)
        print(index)
        #get the single ts
        ts = data_example.groupby('date')['units'].sum()
        
                
        #####COMPUTE RESULTS INCLUSIVE WEEKEND VALUES#######################
        ##fill in missing values
        if agg_level == 'D':
            inclusion_list = [True, False]
        else: inclusion_list = [True]
        
        for inclusion in inclusion_list:
            
            ##################fill in missing data#############################
            df0 = pre_process_fill_absent_values(ts, agg_level, inclusion)
                   
            ###############handling outliers####################
            number_outliers, percentage_outiers, upper, lower, weekday, df1 = pre_process_spikes(copy.deepcopy(df0), agg_level = agg_level, smooth_type = smooth_type)
                

            row_outliers = {index:[number_outliers, percentage_outiers, upper, lower, weekday]}
            row_outliers = pd.DataFrame(row_outliers).T
            df_outliers = df_outliers.append(row_outliers)
                
                
            ################Check the stationarty######################################
            stationarity_status =  test_stationarity(df1.units)
            ################Compute result for weekend invlusie time series############
            
            (arma_mdl, pe_arma, acc_pi_arma, avg_diff_pi_arma,
            pe_prophet, acc_pi_prophet, avg_diff_pi_prophet,
            pe_MTM, acc_pi_MTM, avg_diff_pi_MTM,
            window_size, pe, acc_pi, avg_diff_pi,
            pe_naive, acc_pi_naive, avg_diff_pi_naive,
            pe_average, acc_pi_average, avg_diff_pi_average) = automation_model_iteration(original_df = df0, smoothed_df = df1, smooth_type = smooth_type, inclusion = inclusion, stationarity = stationarity_status)
            
            row = {index: [stationarity_status, arma_mdl, pe_arma, acc_pi_arma, avg_diff_pi_arma,
            pe_prophet, acc_pi_prophet, avg_diff_pi_prophet,
            pe_MTM, acc_pi_MTM, avg_diff_pi_MTM,
            window_size, pe, acc_pi, avg_diff_pi,
            pe_naive, acc_pi_naive, avg_diff_pi_naive,
            pe_average, acc_pi_average, avg_diff_pi_average]}
            
            row = pd.DataFrame(row).T
            
            if inclusion == True:
                df_return_inclusive = df_return_inclusive.append(row)
            else:
                df_return_exclusive = df_return_exclusive.append(row)
            
               
    df_return_inclusive.columns = ['stationarity_status', 'best_order_arma', 'pe_arma', 'acc_pi_arma', 'avg_diff_pi_arma',
            'pe_prophet', 'acc_pi_prophet', 'avg_diff_pi_prophet',
            'pe_MTM', 'acc_pi_MTM', 'avg_diff_pi_MTM',
             'window_size', 'pe_MTM_RW', 'acc_pi_MTM_RW', 'avg_diff_pi_MTM_RW',
            'pe_naive', 'acc_pi_naive', 'avg_diff_pi_naive',
            'pe_average', 'acc_pi_average', 'avg_diff_pi_average']
    if df_return_exclusive.empty:  
        pass
    else:
        df_return_exclusive.columns = df_return_inclusive.columns
        
    df_outliers.columns = ['number_outliers', 'percentage_outiers', 'upper', 'lower', 'weekday_most_spikes']
    return df_return_inclusive, df_return_exclusive, df_outliers
   
  










####################Inputs: orignal dataframe, smoothed dataframe.
####################smooth_type: whether the data has been smoothed################
####################inclusion: whether to include or exclusde weekends data####
####################stationarity: the stationariyt status of the dataframe#####   
def automation_single(index_list, data_grouped, agg_level = 'D', smooth_type = 'normal', model = 'MTM_slides_window'):
    import warnings
    warnings.filterwarnings("ignore")
    from functions import test_stationarity
    from func_pre_process import pre_process_ts_bin, pre_process_spikes, pre_process_fill_absent_values
    import copy
    import numpy as np
    import pandas as pd
    from func_state_space_models import state_space_SARIMAX, state_space_UC
    #from func_prophet_model import automation_prophet_model
    
            
    df_return_inclusive  = pd.DataFrame()
    df_return_exclusive = pd.DataFrame()
  
    #do the iteration for every group in the filtered dataset.
    for index in index_list:
        index = tuple(index)
        data_example = data_grouped.get_group(index)
        print(index)
        #get the single ts
        ts = data_example.groupby('date')['units'].sum()
        ts = np.ceil(ts)

        #####COMPUTE RESULTS INCLUSIVE WEEKEND VALUES#######################
        ##fill in missing values
        if agg_level == 'D':
            inclusion_list = [True, False]
        else: inclusion_list = [True]
        
        for inclusion in inclusion_list:
            ##################fill in missing data#############################
            df0 = pre_process_fill_absent_values(ts, agg_level, inclusion)
                   
            ###############calculate the percentage of outliers####################
            number_outliers, percentage_outiers, upper, lower, weekday, df1 = pre_process_spikes(copy.deepcopy(df0), agg_level = agg_level, smooth_type = smooth_type)
 
            ################Check the stationarty######################################
            stationarity_status =  test_stationarity(df1.units)
            
            ###############do the prediction using the MTM model##################
            #df_return_inclusive.columns = ['pe', 'acc_pi', 'avg_diff_pi']       
            if model == 'state_space_model':
                pe, acc_pi, avg_diff_pi = state_space_SARIMAX(df1)
                pe_, acc_pi_, avg_diff_pi_ = state_space_UC(df1)
                
                row = {index: [pe, acc_pi, avg_diff_pi, pe_, acc_pi_, avg_diff_pi_]}
                row = pd.DataFrame(row).T
                
                if inclusion == True:
                    df_return_inclusive = df_return_inclusive.append(row)
                else:
                    df_return_exclusive = df_return_exclusive.append(row)
            else:# model == 'Prophet'  
                for changepoint_number in [0, 5, 10, 20, 25, 50]:
                    for changepoint_scale in [0.005, 0.05, 0.5]:
                        prophet_mdl, pe, acc_pi, avg_diff_pi = automation_prophet_model(df0, df1, smooth_type, changepoint_number, changepoint_scale, inclusion)
                        row = {index: [tuple([changepoint_number, changepoint_scale]), pe, acc_pi, avg_diff_pi]}
                        row = pd.DataFrame(row).T
                        
                        if inclusion == True:
                            df_return_inclusive = df_return_inclusive.append(row)
                        else:
                            df_return_exclusive = df_return_exclusive.append(row)
                
                #df_return_inclusive.columns = ['changepoint_number_scale', 'pe', 'acc_pi', 'avg_diff_pi']
            
    if len(df_return_inclusive.columns) == 3: 
        df_return_inclusive.columns = ['pe', 'acc_pi', 'avg_diff_pi']
    elif len(df_return_inclusive.columns) == 4:
        df_return_inclusive.columns = ['window_size','pe', 'acc_pi', 'avg_diff_pi']
    else:
        df_return_inclusive.columns = ['pe_SARIMAX', 'acc_pi_SARIMAX', 'avg_diff_pi_SARIMAX', 'pe_UC', 'acc_pi_UC', 'avg_diff_pi_UC']
        
        
    if df_return_exclusive.empty:  
        pass
    else:
        df_return_exclusive.columns = df_return_inclusive.columns
    
            
    return df_return_inclusive, df_return_exclusive
     
