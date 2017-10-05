# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 11:00:11 2017

"""

import os
import pandas as pd
import numpy as np

def find_suffix_filenames(path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

    
def import_data(data_folder = './data'):
    filenames  = find_suffix_filenames(data_folder, '.csv')
    NEWFILE = True
    for filename in filenames:
        print filename
        data_new = pd.read_csv(os.path.join(data_folder,filename))
        if NEWFILE:
            data = data_new
        else:
            data = pd.concat([data, data_new], ignore_index=True)
    
    
    COLS_DATETIME =['Created Date','Closed Date','Resolution Action Updated Date']
    for col in COLS_DATETIME:
        data[col] = pd.to_datetime(data[col])
    
    data.sort_values(by = 'Created Date', ascending = True, inplace = True, kind ='mergesort')
    
    data = data.dropna(subset = ['Created Date']).drop_duplicates()
    #actually no NA in created date and no duplicates
    
    # drop columns with significant NAs > 50% for now
    data.replace({'Unspecified':np.nan}, inplace = True)
    
    summary_nas =  (data.isnull().sum()*1.0/data.shape[0])
    data = data[summary_nas[summary_nas < 0.5].index]
    
    # drop some pre-defined columns
    COLS_TO_DROP = ['Community Board'] 
    data.drop(COLS_TO_DROP, axis = 1, inplace =True)
        
    return data

def summary_data(data):
    summary_nas = (data.isnull().sum()*1.0/data.shape[0])
    

    
    summary_data = {'NAs': summary_nas}
    
            
    
    return summary_data
         