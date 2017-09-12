# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 15:00:01 2017

@author: jdkern
"""

from __future__ import division
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import scipy.stats as stats

df_temps=pd.read_csv('tempdata.csv',header=None)
df_temps.columns = ('Date','Temp')
temps = df_temps.loc[:,'Temp'].as_matrix().astype(np.float)

#Read electricity demand data
df_demand = pd.read_csv('hourly-day-ahead-bid-data-2014.csv',header=4)
# get rid of 'date' column in data
del df_demand['Date']
demand = df_demand.as_matrix().astype(np.float)

#convert to a vector
def mat2vec(data):
    [rows,columns] = np.shape(data)
    vector = []
    for i in range(0,rows):
        vector = np.append(vector,data[i,:])
     
    return vector

vector_demand = mat2vec(demand)

#convert to peak demand vector
peaks = []

for i in range(0,365):
    peak_hourly = np.max(demand[i,:])
    peaks = np.append(peaks,peak_hourly)

#peaks = peaks/1000

# forms 2-column matrix
combined = np.column_stack((temps,peaks))

#look for NaNs
for i in range(0,len(combined)):
    if np.isnan(combined[i,1]) > 0:
        combined[i,1] = np.mean([combined[i-1,1],combined[i+1,1]])
        
#clusters for each row
IDX = KMeans(n_clusters=3, random_state=0).fit_predict(combined)

#forms 3-column matrix
clustered_data = np.column_stack((combined,IDX))


plt.figure()
plt.scatter(combined[:,0],combined[:,1],c=IDX+1)
plt.xlabel('Temps (F)',fontsize=24)
plt.ylabel('Electricity Demand (MWh)',fontsize=24)




#plotting average hourly demand profile for Jan
Jan_Avg = np.zeros((24,1))

for i in range(0,24):
   
   Jan_Avg[i] = np.mean(demand[0:31,i])
   
plt.figure()
hours = np.arange(1,25)
plt.scatter(hours,Jan_Avg)


#plotting average hourly demand profile for July
Jul_Avg = np.zeros((24,1))

for i in range(0,24):

   Jul_Avg[i] = np.mean(demand[181:212,i])
   
plt.figure()
hours = np.arange(1,25)
plt.scatter(hours,Jul_Avg)


#Jan 1st was Wed so give it a 4


a = np.array([4,5,6,7,1,2,3])
b = np.tile(a,52)
day_index = np.append(b,4)
final = np.column_stack((day_index,combined[:,1]))
#final is the peak daily electricity matrix with each day of 2014 indexed by day of the week
#sorting final matrix by first column (days of week)
final_sorted = final[final[:,0].argsort()]
#create a boxplot showing the mean and interquartile ranges for each day of the week
data = [final_sorted[0:51,1],final_sorted[52:103,1],final_sorted[104:155,1],final_sorted[156:208,1],final_sorted[209:260,1],final_sorted[261:312,1],final_sorted[313:364,1]]

plt.figure()
plt.boxplot(data)


plt.xlabel('Days of the Week')
plt.ylabel('Variance for Peak Daily Electricity Demand by Day of the Week 2014')




    

    

    

