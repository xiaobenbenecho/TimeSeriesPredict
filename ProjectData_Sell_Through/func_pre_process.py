# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:18:28 2019

@author: yzhang
"""




####################inclusion means whether we will include the weekends sales data##
#inputs: a time series ts; a Boolean value indicats whetehr weekends data is included. 
def pre_process_fill_absent_values(ts, agg_level = 'D', inclusion = True):
    import datetime
    import pandas as pd
    
    ####################fill in missing values#################################
    if agg_level == 'W':
        index = pd.date_range(start = ts.index.min(), end = ts.index.max(), freq = agg_level) - datetime.timedelta(days=6)
    else:
        index = pd.date_range(start = ts.index.min(), end = ts.index.max(), freq = agg_level) 
        
    ts_ =  pd.DataFrame(index = index)
    #left join the ts_
    ts = ts_.merge(right = ts.to_frame(), right_index= True, left_index = True, how = 'left')
    del(ts_)
    ts = ts.T.squeeze()
    
    
    ##########interplot the missing values#####################################
    ts = pd.DataFrame(ts)
    ts = ts.fillna(0)
    
    
    if (inclusion == False):
        ts['weekday'] = ts.index.strftime("%A")
        ts = ts.loc[ts['weekday'].isin(['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
        ts = ts.drop(['weekday'], axis = 1)
        
    return ts
    


################Seperate the sales units into bins predict the indicator of the bin in order to smooth the time series######
################input is the dataframe after filling the missing and absent data
#df = df0
def pre_process_ts_bin(df):
    import pandas as pd
    import numpy as np
    from math import ceil
    
    number_quantile = min(ceil(len(df)/4), 100)
    
    bins = np.unique(np.quantile(df.iloc[:, 0], np.arange(0, number_quantile+1)/number_quantile)) 
    
    bins[0] = bins[0]-0.1
    group_names = []
    for i in range(0,len(bins)-1):
        group_names.append((bins[i] + bins[i+1])/2)

    df.units = (pd.cut(df.units, bins, labels = group_names)).astype(int)
    
    return df, bins, group_names


################Find, count and replace the outliers######
################input is the dataframe after filling the missing and absent data

def pre_process_spikes(df0, agg_level = 'D', smooth_type = 'outliers_interpolated_mean'):
    from functions import outliers_iqr
    import copy
    import numpy as np
    
    
    ###############calcualte the upper and lower bound############
    ##calculate the boundary of upper and lower boundary
    quartile_1, quartile_3 = np.percentile(df0, [25, 75])
    iqr = quartile_3 - quartile_1
    upper = quartile_3 + (iqr * 1.5)
    lower = quartile_1 - (iqr * 1.5)
    ###############detect the spikes##############################
    #use 1.5IQR to detect the outliers.
    #Outliners? and how to deal with the outliners
    outliers =  df0.loc[outliers_iqr(df0),:]
    
    if len(outliers) == 0:
        weekday = 'NoOutliers'
        percentage_outliers  = 0
    else:
        if agg_level == 'D':
            #weekdays of the outliers: all of them happen on weekdays, Monday the most frequent day, this may due to the saled unites during weekdays are acumulated on Monday.
            weekday = outliers.index.strftime("%A").value_counts().index[0]
            #######Weekly aggregated level.
        else:# agg_level == 'W':
            ############Return the week number of the year#####################
            weekday = outliers.index.strftime('%W').value_counts().index[0]
            
    #count the spikes
    #calcualte the percentage of outliers
    percentage_outliers  = len(outliers)/len(df0)
    #replace the outliers with average of the data
    if smooth_type == 'outliers_interpolated_mean':
        value = int(df0[df0.columns[0]].mean())
        df0.loc[outliers.index, df0.columns[0]] = value
    elif smooth_type == 'outliers_interpolated_iqr':
        col = df0.columns[0]
        df0[col] = np.where(df0[col] > upper, upper, df0[col])
        df0[col] = np.where(df0[col] < lower, lower, df0[col])
    else: # smooth_type == 'normal'
        pass

    return len(outliers), percentage_outliers, upper, lower, weekday, df0



