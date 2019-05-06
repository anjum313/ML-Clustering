"""
Created on Mon Feb 19 20:05:01 2018

@author: anjum
"""
# Harirical  Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data set
dataset =pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:, [3,4]].values

# Useing the dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method ='ward'))
plt.title('Dendograms')
plt.xlabel('Coustomers')
plt.ylabel('Eucladian Distances')
plt.show()

# Fitting hierarchical clustering to the dataset
Number_clusters = 5
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters =5, affinity= 'euclidean',linkage='ward')
y_hc = hc.fit_predict(X)


# Visvulaizing the results.
plt.scatter(X[y_hc == 0 ,0], X[y_hc == 0 ,1], s = 100 ,c ='red', label = 'Careful')    
plt.scatter(X[y_hc == 1 ,0], X[y_hc == 1 ,1], s = 100 ,c ='blue', label = 'Standard')  
plt.scatter(X[y_hc == 2 ,0], X[y_hc == 2 ,1], s = 100 ,c ='green', label = 'Targer high incom  high spending')  
plt.scatter(X[y_hc == 3 ,0], X[y_hc == 3 ,1], s = 100 ,c ='cyan', label = 'CCareless')  
plt.scatter(X[y_hc == 4 ,0], X[y_hc == 4 ,1], s = 100 ,c ='magenta', label = 'Sensable')
plt.title('cluster of clints')
plt.xlabel('Annaual Incom (KS)')
plt.ylabel('Spending Score (1-100)') 
plt.legend()
plt.show()  