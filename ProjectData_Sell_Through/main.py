# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:07:28 2019

@author: yzhang
"""

#The main function

######GLOBAL variables#########################################################
############agg_level: 'W'- weekly aggregated level 'D'- daily aggregated level
agg_def = ['country', 'vendor', 'product_range']
agg_level = 'W'
smooth_type_list = ['normal', 'outliers_interpolated_mean', 'outliers_interpolated_iqr']

################import the data################################################
import pandas as pd
data = pd.read_csv('C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\NewData\\SELL_THROUGH_PC.csv')
data.columns

##############Keep only the useful rows########################################
cols = ['date', 'country', 'vendor_id_scramble', 'product_range_id_scramble', 'product_id_scramble', 'units', 'revenue']
data = data[cols]
data = data.dropna(axis = 0)
data.columns = ['date', 'country', 'vendor','product_range', 'product', 'units', 'revenue']


##############Filter the feasible products#####################################
from functions import *
summary_data, data_grouped, total_number_product = filter_feasible_product(data
      = data, agg_level = agg_level, agg_def = agg_def)


#######################Iteration of all feasible products on the univariate models########

################iterate all models on all feasible products############################

#temp1 = summary_data.head(10)
from func_automation import *
for smooth_type in smooth_type_list:
    
    df_inclusive, df_exclusive, df_outliers = automation(summary_data.index, data_grouped, agg_level = agg_level, smooth_type = smooth_type)
    
    ########store the data######
    inclusion_file_name = 'df_inclusive_' + smooth_type + '_' + agg_level
    exclusion_file_name = 'df_exclusive_' + smooth_type + '_' + agg_level
    outlier_file_name = 'df_outliers_' +  smooth_type + '_' + agg_level
    
    
    df_inclusive.to_pickle('C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\ResultsProjectSellThroughPC\\' + inclusion_file_name + '.pkl')
    df_exclusive.to_pickle('C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\ResultsProjectSellThroughPC\\' + exclusion_file_name + '.pkl')
    df_outliers.to_pickle('C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\ResultsProjectSellThroughPC\\' + outlier_file_name + '.pkl')


pe_cols = [col for col in df_inclusive.columns if 'pe' in col]

temp = df_inclusive[pe_cols]
temp = temp.apply(pd.to_numeric)
temp.mean()

#############################iterate for just one model#################################
model = 'state_space_model'
from func_automation import *

#temp1 = summary_data.head(10)
for smooth_type in smooth_type_list:
    
    df_inclusive, df_exclusive = automation_single(summary_data.index, data_grouped, agg_level = agg_level, smooth_type = smooth_type, model = model)
    
    ########store the data######
    inclusion_file_name = 'sample_df_inclusive_' + smooth_type + '_' + agg_level + '_' + model
    exclusion_file_name = 'sample_df_exclusive_' + smooth_type + '_' + agg_level+ '_' + model
    #outlier_file_name = 'sample_df_outliers_' +  smooth_type + '_' + agg_level+ '_'+ model
    
    
    df_inclusive.to_pickle('C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\ResultsProjectSellThroughPC\\' + inclusion_file_name + '.pkl')
    df_exclusive.to_pickle('C:\\Users\\yzhang\\Desktop\\KTPAssociate\\Data\\ResultsProjectSellThroughPC\\' + exclusion_file_name + '.pkl')


   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##############Distributin of the products according to noDays##################
##############Majority of the products have observations less than 100#########
from EDA import *
df = aggregation(data, agg_def)
import matplotlib.pyplot as plt
plt.hist(df.noDays, bins = 'auto') 
plt.xlabel("Number of days with sales records")
plt.ylabel('Count Number of Products')
#plt.xlim(left =  100)
#plt.ylim(top = 2000)


###########################EDA#################################################
# check whehte the selected products are Top sellers for considered vendor or vendor and country
summary_country_vendor = top_seller_by_country_vendor(data, summary_data).sort_values('ratio', ascending = False)
summary_vendor = top_seller_by_country_vendor(data, summary_data, ['vendor'])

import seaborn as sns
sns.distplot(summary_country_vendor['ratio'], kde = False)
sns.distplot(summary_vendor['ratio'], kde = False)

#summary of the products according to number of records.
print ("The total number of time series is : " + str(len(df)))
print('The percentage of feasible time series is: ' + "{0:.2%}".format(len(summary_data)/len(df)) )


######sales units on each weekday
summary_weekday = units_on_weekday_or_month(data, agg_type = 'weekday')

######sales units on each month
summary_month = units_on_weekday_or_month(data, agg_type  = 'month')

#####draw the counts showing the distribution of the sales units on each weekday/month
import numpy as np
g = sns.catplot(x="weekday", col="country", data =  summary_weekday, kind= "count", 
                legend = True, col_wrap = 3)

h = sns.catplot(x="month", col="country", data =  summary_month, kind= "count", 
                legend = True, col_wrap = 3)

##########Count of Absent values on each Weekday###############################

####for all products###########################################################
number_0sales =  nonexistent_values_weekday(data)
print(number_0sales)
number_0sales.plot.bar()
plt.ylabel('Number of days with 0 sales')
plt.xlabel('Weekday')
plt.show()


####for the feasible products##################################################
filter_data = pd.DataFrame()
for index in summary_data.index:
    index = tuple(index)
    data_example = data_grouped.get_group(index)
    filter_data = filter_data.append(data_example)

number_0sales =  nonexistent_values_weekday(filter_data)
print(number_0sales)
number_0sales.plot.bar()
plt.ylabel('Number of days with 0 sales')
plt.xlabel('Weekday')
plt.show()

   