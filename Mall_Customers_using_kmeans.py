# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:30:35 2024

@author: Nazib
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Mall_Customers.csv')
df.head()
df.info()
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
from sklearn.preprocessing import MinMaxScaler

# Feature normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
print(wcss)

plt.figure(figsize=(10, 6)) 
plt.plot(range(2, 11), wcss, marker='o', linestyle='--', color='b')  
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, 11))  
plt.grid(True) 
plt.show()


nc=6
kmeans = KMeans(n_clusters=nc)
kmeans.fit(X_scaled)
plt.figure(figsize=(15, 8))
for cluster_label in range(nc):  # Loop through each cluster label
    cluster_points = X[kmeans.labels_ == cluster_label]
    centroid = cluster_points.mean(axis=0)  # Calculate the centroid as the mean position of the data points
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],
                s=50, label=f'Cluster {cluster_label + 1}')  # Plot points for the current cluster
    plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')  # Plot the centroid
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()