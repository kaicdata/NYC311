# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 10:57:04 2017
"""

import pandas as pd
import numpy as np
import path
import os
import time
import datetime as dt
import matplotlib.pyplot as plt
import common as com
#from networkx.drawing.nx_agraph import graphviz_layout

ROOT_DIR = "C:\\Disks\\D\\Coatue\\NYC311\src"
#ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(ROOT_DIR)

data = com.import_data(existing = True)
data['ZipDate'] = data['Created Day'].map(str)+'_'+data['Incident Zip'].map(str)


summary_data = com.summary_data(data)

data_other_byzip = com.import_data_other()

print "saving to local"
data.to_csv('./output/data_nyc311.csv', index = False)


#%%
## GEO EDA:
"""
#(1)
data_byTypeZip = data.groupby(by = ['Complaint Type','Incident Zip'])['Unique Key'].count().reset_index()
data_byTypeZip = pd.merge(data_byTypeZip, data_other_byzip[['zipcode','population','people per sq mile','NumOfBusiness']], left_on = 'Incident Zip', right_on ='zipcode', how = 'inner')

summary_CorrToPopulation = data_byTypeZip.groupby(by='Complaint Type')['Unique Key','population'].corr(min_periods =5).xs('Unique Key', level =1).loc[:,['population']].dropna().sort_values(by ='population',ascending = False)
summary_CorrToBusiness = data_byTypeZip.groupby('Complaint Type')['Unique Key','NumOfBusiness'].corr(min_periods =5).xs('Unique Key', level =1).loc[:,['NumOfBusiness']].dropna().sort_values(by ='NumOfBusiness',ascending = False)

summary_CorrToPopulation.head(5)
summary_CorrToBusiness.head(5)

# http://www.oneonta.edu/faculty/vomsaaw/w/psy220/files/SignifOfCorrelations.htm for how significant population plays a role here
data[data['Complaint Type'] == 'Electronics Waste'].shape[0]*1.0/data.shape[0]
"""

#%%
'''
#################################
#################################


Selection I -- Occurrence Definition 
AND Complaint Type Network Analysis


#################################
#################################

'''

'''

Section I.1 - Occurrence - Pairwide Mutual Information (PMI) on complaint types

i.e. low PMI pairs are treated as more independent pairs
'''

#%% Complaint Type Clustering and Feature Selection

## First include all complaint types

data_sub1 = data.dropna(subset = ['Incident Zip']) 
data_sub1['ZipDate'] = data_sub1['Created Day'].map(str)+'_'+data_sub1['Incident Zip'].map(str)
data_sub1['ZipDate'] = [ll.split('.')[0] for ll in data_sub1.ZipDate]
data_ZipDate = data_sub1.groupby(['Complaint Type','ZipDate'])['Unique Key'].count().reset_index()
data_ZipDate = pd.pivot_table(data_ZipDate, index = 'Complaint Type', columns =['ZipDate'], values = 'Unique Key', aggfunc = sum)


#%%
# Initial feature selection using Pairwire Mutial Information to exclude statndalone events

#### PMI feature
PMI_all = com.PMI_NPMI(data_ZipDate)
PMI = PMI_all.loc[PMI_all.NPMI>0.2]
PMI = PMI.sort_values('NPMI', ascending = False)

PMI.to_csv('./output/PMI.csv',index=False)

print PMI.shape


'''

Section I.2 - Network Analysis on PMI selected complaint types

i.e. select pairs with relatively high PMI and construct network on top of this. edge weights are the PMI values.

'''
## Using network analysis concept to exclude/include events with broader network impacts

import networkx as nx

G=nx.from_pandas_dataframe(PMI.head(300), 'Complaint_Type1', 'Complaint_Type2', ['NPMI'])
#nx.draw(G,with_labels=True)
#plt.draw()
#plt.show()

#centrality measures to look at importance of event nodes
centrality_G = com.centrality_measures(G)

## find separable connected component subgraphs
component_subgraphs = nx.connected_component_subgraphs(G)
x=[len(c) for c in component_subgraphs]
print x

## pick the subgraph(s) with significant number of events -- i.e. events that can cooccur with only very small number of other events are of less interests
# in this case pick the subgroup with 64 nodes
component_subgraphs = nx.connected_component_subgraphs(G)
subgraph = list(component_subgraphs)[0]


# get the centrality measures
centrality_subgraph = com.centrality_measures(subgraph)

# show the largest subgraphc
#s1 = list(centrality_subgraph.loc[centrality_subgraph['Betweenness Centrality']>0.4, 'Complaint Type'])
#s2 = list(centrality_subgraph.loc[(centrality_subgraph['Betweenness Centrality']>0)&(centrality_subgraph['Betweenness Centrality']<=0.4), 'Complaint Type'])
#s3 = list(centrality_subgraph.loc[centrality_subgraph['Betweenness Centrality']<=0, 'Complaint Type'])

nx.draw(subgraph, pos=nx.spring_layout(subgraph),node_size =100, with_labels = True)
#nx.draw_networkx_nodes(subgraph, pos=nx.spring_layout(subgraph), node_size = 250, node_color = 'g', with_labels = True, nodelist = s1)
#nx.draw_networkx_nodes(subgraph, pos=nx.spring_layout(subgraph), node_size = 160, node_color = 'b', label = s2, nodelist = s2)
#nx.draw_networkx_nodes(subgraph, pos=nx.spring_layout(subgraph), node_size = 100, node_color = 'r', label = s3, nodelist = s3)



nx.draw(subgraph,with_labels=True, node_size=160)
plt.savefig("./output/subgraph_select.png")
nx.draw(subgraph,with_labels=False, node_size=100)
plt.savefig("./output/subgraph_select_nolabel.png")
plt.show()


## save graphc to gexf for gephi visualization
# Use 1.2draft so you do not get a deprecated warning in Gelphi
#nx.write_gexf(subgraph, './output/selected_subgraph.gexf',encoding='utf-8', prettyprint=True, version='1.2draft')
#nx.write_gml(subgraph, './output/selected_subgraph.gml')

#%%
"""

only select out the subgraph with more nodes conected - most subgraphs have only 2-5 nodes, which are of less interest in this analysis

"""



#%%
'''
#################################
#################################


Selection II -- Clustering Days based on complaint occurrences on selected types. By zipcode and days

i.e. find days that are similar to historical days in terms complaint patterns. 

Used model derived from tf-idf document clustering methodology.

#################################
#################################

'''

#%% Clustering
## dimentiona 1  --time clustering: find days with similar historical days
# kmeans on tf-idf  

data_subset = data.dropna(subset = ['Incident Zip'])
data_subset = pd.merge(data_subset, pd.DataFrame({'Selected Type': subgraph.nodes()}), left_on = 'Complaint Type', right_on ='Selected Type', how ='inner')
#data_subset = pd.merge(data_subset, pd.DataFrame({'Selected Type':centrality_subgraph.loc[centrality_subgraph['Betweenness Centrality']>0.1, 'Complaint Type']}), left_on = 'Complaint Type', right_on ='Selected Type', how ='inner')


data_subset['ZipDate'] = data_subset['Created Day'].map(str)+'_'+data_subset['Incident Zip'].map(str)
data_subset['ZipDate'] = [ll.split('.')[0] for ll in data_subset.ZipDate]
data_subset_ZipDate = data_subset.groupby(['Complaint Type','ZipDate'])['Unique Key'].count().reset_index()
data_subset_ZipDate = pd.pivot_table(data_subset_ZipDate, index = 'Complaint Type', columns =['ZipDate'], values = 'Unique Key', aggfunc = sum)

plt.imshow(data_subset_ZipDate.ix[:,1:1000], cmap='hot', interpolation='nearest')
plt.show()

data_subset_ZipDate_idf = np.log(data_subset_ZipDate.shape[1]*1.0/((data_subset_ZipDate.fillna(0))>0).sum(axis=1))

data_subset_ZipDate_tfidf = data_subset_ZipDate.multiply(data_subset_ZipDate_idf,axis="index").fillna(0)



### input data matrix to analysis

data_tomodel = data_subset_ZipDate_tfidf.transpose()
data_training = data_tomodel.iloc[:int(np.floor(data_tomodel.shape[0]*0.8))]
data_testing = data_tomodel.iloc[int(np.floor(data_tomodel.shape[0]*0.8)):]



#%% Clustering days using kmeans 
## use eblow method to choose k-means
com.eblow(data_training, n_max = 100)

# chose N =30 from eblow method

from sklearn.cluster import KMeans

num_clusters = 40

km = KMeans(n_clusters=num_clusters)

km.fit(data_training)

clusters = km.labels_.tolist()

##  predict
data_training['cluster_label'] = clusters
data_testing['cluster_label'] = km.predict(data_testing)
data_training['training'] =1
data_testing['training'] = 0

data_tomodel = pd.concat([data_training, data_testing], ignore_index=False)

data_output_reducedfeatures = data_subset_ZipDate[data_tomodel.index].transpose().fillna(0)
data_output_reducedfeatures = data_output_reducedfeatures.join(data_tomodel[['cluster_label','training']])


data_output_allfeatures = data_ZipDate[data_tomodel.index].transpose().fillna(0)
data_output_allfeatures = data_output_allfeatures.join(data_tomodel[['cluster_label','training']])

# 
data_output_reducedfeatures['# of active complaint types'] = ((data_output_reducedfeatures.drop(['training','cluster_label'], axis=1))>0).sum(axis =1)
data_output_allfeatures['# of active complaint types'] = ((data_output_allfeatures.drop(['training','cluster_label'], axis =1))>0).sum(axis =1)

## save output
data_output_reducedfeatures.to_csv('./output/NYC311_2014to2017_reducedfeatures.csv', index=True)
data_output_allfeatures.to_csv('./output/NYC311_2014to2017_allfeatures.csv', index=True)

#
##I use joblib.dump to pickle the model, once it has converged and to reload the model/reassign the labels as the clusters.
from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

joblib.dump(km,  './output/doc_cluster.pkl')

km = joblib.load('./output/doc_cluster.pkl')
clusters = km.labels_.tolist()



## cluster summary

summary_training = data_training.groupby('cluster_label')['training'].count().sort_values(ascending = False)
summary_testing = data_testing.groupby('cluster_label')['training'].count().sort_values(ascending = False)



#%%
'''
#################################
#################################


Selection III -- TOPIC MODELING -- Latent Dirichelet Allocation


#################################
#################################

'''

G=nx.from_pandas_dataframe(PMI.head(350), 'Complaint_Type1', 'Complaint_Type2', ['NPMI'])
component_subgraphs = nx.connected_component_subgraphs(G)
subgraph = list(component_subgraphs)[0]
 
data_LDA = pd.merge(data, pd.DataFrame({'Selected Type': subgraph.nodes()}), left_on = 'Complaint Type', right_on ='Selected Type', how ='inner')
#data_LDA = pd.merge(data, pd.DataFrame({'Selected Type': list(PMI['Complaint_Type1'].unique())}), left_on = 'Complaint Type', right_on ='Selected Type', how ='inner')
#data_LDA = pd.merge(data, pd.DataFrame({'Selected Type': list(PMI['Complaint_Type1'].head(350).unique())}), left_on = 'Complaint Type', right_on ='Selected Type', how ='inner')


data_LDA_ZipDate = data_LDA.groupby(['Complaint Type','ZipDate'])['Unique Key'].count().reset_index()
data_LDA_ZipDate = pd.pivot_table(data_LDA_ZipDate, index = 'Complaint Type', columns =['ZipDate'], values = 'Unique Key', aggfunc = sum)



# construct the texts
from gensim import corpora, models, similarities
texts = [] 
docs = []
count = 0
for col in data_LDA_ZipDate.columns:
    col_text = data_LDA_ZipDate[col]
    tokens = list(col_text[~pd.isnull(col_text)].index)
    if tokens:
        texts.append(list(col_text[~pd.isnull(col_text)].index))
        docs.append(col)
    count = count+1
    if count%10000.0 == 0:
        print count


# #create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)
#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

corpus_training= corpus[:int(np.floor(len(corpus)*0.8))]
corpus_testing = corpus[int(np.floor(len(corpus)*0.8)):]


# show the topics
log_perplexity = {'num of topics':[], 'log_perplexity (oob)':[]}
# determing number of topics to use
for num in range(30,200,20):
    # fit a LDA model
    print num
    lda = models.LdaModel(corpus_training, num_topics=num, 
                            id2word=dictionary)
    log_perplexity['num of topics'].append(num)                                                
    log_perplexity['log_perplexity (oob)'].append(lda.log_perplexity(corpus_testing))                        

log_perplexity = pd.DataFrame(log_perplexity)
# takes less than five minutes to run LDA


# choose 
lda = models.LdaModel(corpus_training, num_topics= 80, 
                            id2word=dictionary) 
   
lda.show_topics(num_topics=80, num_words=3)
# calculate log perplexity to 
lda.log_perplexity(corpus)
# get term topics
topic_testing =[]
for test in corpus_testing:
    topic_testing.append(lda.get_document_topics(test))
    



