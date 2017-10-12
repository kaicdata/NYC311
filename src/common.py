# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 11:00:11 2017

"""

import os
import pandas as pd
import numpy as np
import datetime as dt
import networkx as nx

import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist

def find_suffix_filenames(path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

    
def import_data(data_folder = './data', existing = True):
    if existing:
        data = pd.read_csv('./output/data_nyc311.csv')
    else:

        filenames  = find_suffix_filenames(data_folder, '.csv')
        NEWFILE = True
        for filename in filenames:
            print filename
            data_new = pd.read_csv(os.path.join(data_folder,filename))
            if NEWFILE:
                data = data_new
                NEWFILE = False
            else:
                data = pd.concat([data, data_new], ignore_index=True)
        
             
        
        # select specific columns
        COLS_TO_SELECT = ['Unique Key', 'Created Date', 'Complaint Type', 'Incident Zip','Borough']        
        data = data[COLS_TO_SELECT]        

        # columns format to datetime
        print "formatting columns"   
        
        COLS_DATETIME =['Created Date']
        for col in COLS_DATETIME:
            data[col] = pd.to_datetime(data[col])
        
        # 
        """
        print "sorting on Created Date"
        data.sort_values(by = 'Created Date', ascending = True, inplace = True, kind ='mergesort')
        
        data = data.dropna(subset = ['Created Date']).drop_duplicates()
        #actually no NA in created date and no duplicates
        """
        # drop columns with significant NAs > 50% for now
        print "replace Unspecified to NaN"
        data.replace({'Unspecified':np.nan}, inplace = True)
        
        # 
        print "Calculating summaries"
        summary_nas =  (data.isnull().sum()*1.0/data.shape[0])
        data = data[summary_nas[summary_nas < 0.5].index]
        
        # drop some pre-defined columns
        #COLS_TO_DROP = ['Community Board','Closed Date','Resolution Action Updated Date','X Coordinate (State Plane)','Y Coordinate (State Plane)','Location','Park Borough','Latitude','Longitude','Resolution Description','Incident Address','Street Name','Cross Street 1','Cross Street 2','Agency Name'] 
        #data.drop(COLS_TO_DROP, axis = 1, inplace =True)
        # add time column
        
        data['Created Day'] = [ll.strftime("%Y-%m-%d") for ll in data['Created Date']] 
        data['Created Time'] = [ll.strftime("%H:%M:%S") for ll in data['Created Date']]        
                
        ## lots of them have Created Time being 00:00:00 so can only work on date level for now - not sure how good the quality is.
        # 2378 out of 54576 records have time being 00:00:00
        
        ## zip code columns if 12345-3451 then return 12345
        data['Incident Zip'] = [str(ll).split('-')[0] for ll in data['Incident Zip']]
        data['Incident Zip'] = [str(ll).split('.')[0] for ll in data['Incident Zip']]

        data.dropna(subset = ['Incident Zip'], inplace=True) 
        data = data.loc[data['Incident Zip']!= 'nan']        
    return data

def import_data_other(data_popbyzip = './data/other/PopulationByZipcode.csv', data_numofbusiness = './data/other/zbp15detail.txt'):
    data_population = pd.read_csv(data_popbyzip)
    data_business = pd.read_csv(data_numofbusiness)
    
    data_business_sum = data_business.drop('naics', axis=1).groupby(by = 'zip').sum().reset_index()
    
    data_other_byzip = pd.merge(data_population, data_business_sum, left_on = 'zipcode', right_on = 'zip', how = 'left')
    
    data_other_byzip = data_other_byzip.rename(columns ={'est':'NumOfBusiness'})
    
    return data_other_byzip

def summary_data(data):
    summary_nas = (data.isnull().sum()*1.0/data.shape[0])
    #summary_city = data.groupby('City', index= False).coun

    
    summary_data = {'NAs': summary_nas}
    
    return summary_data
 

def PMI_NPMI(data_ZipDate):
    # PMI between different complaint types
    PMI ={'Complaint_Type1':[], 'Complaint_Type2':[], 'PMI':[], 'NPMI':[]}
    C_x = ((data_ZipDate.fillna(0))>0).sum(axis=1)
    N = data_ZipDate.shape[1]
    for type1 in data_ZipDate.index:
        print type1
        for type2 in data_ZipDate.index:
            if type1 != type2:
                p_xy = ((((data_ZipDate.loc[type1].fillna(0))>0)*((data_ZipDate.loc[type2].fillna(0))>0)).sum())*1.0/N
                p_x = C_x.loc[type1]*1.0/N
                p_y = C_x.loc[type2]*1.0/N
                PMI['Complaint_Type1'].append(type1)
                PMI['Complaint_Type2'].append(type2)
                PMI['PMI'].append(np.log(p_xy)-np.log(p_x)-np.log(p_y))
                PMI['NPMI'].append((np.log(p_x)+np.log(p_y))/np.log(p_xy)-1)
    
    return pd.DataFrame(PMI) 
   
# eblow methods to select kmeans cluster
def eblow(df, n_max = 50, n_min =10):
    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in range(n_min, n_max,5)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2)/df.values.shape[0]
    bss = tss - wcss
    plt.plot(range(n_min, n_max,5),bss*1.0/tss)
    plt.ylabel('# of Total Variance Explained')
    plt.xlabel('# of Clusters')
    plt.savefig("./output/eblow_output.png")
    plt.show()

    
## dnetwork analysis funcds

def sorted_map(map):
    ms = sorted(map.iteritems(),key=lambda (k,v): (-v,k))
    return ms
    
def centrality_measures(G):
    degree = nx.degree(G)
    degree_sorted = pd.DataFrame(sorted_map(degree))
    degree_sorted.columns = ['Complaint Type', 'Degree Centrality']
    
        
    # closeness centrality
    cs = nx.closeness_centrality(G)    
    cs_sorted = pd.DataFrame(sorted_map(cs))
    cs_sorted.columns =['Complaint Type', 'Closeness Centrality']
    
    # betweenness centrality
    b = nx.betweenness_centrality(G)
    b_sorted = pd.DataFrame(sorted_map(b))
    b_sorted.columns = ['Complaint Type', 'Betweenness Centrality']
    
    # eigenvector centrality
    try:    
        eigen_central = nx.eigenvector_centrality(G)
        eigen_central_sorted = pd.DataFrame(sorted_map(eigen_central))
        eigen_central_sorted.columns = ['Complaint Type','Eigenvector Centrality']
    
        centrality = b_sorted.merge(cs_sorted, on='Complaint Type', how ='left').merge(eigen_central_sorted, on='Complaint Type', how ='left').merge(degree_sorted, on='Complaint Type', how ='left')
    except:
        centrality = b_sorted.merge(cs_sorted, how ='left').merge(degree_sorted, on='Complaint Type', how ='left')        
    return centrality