import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Dataset
df = pd.read_csv('House Prices/train.csv')
X_train = df[['LotArea']]

#K-means Clustering
numClusters = 3 #Change this as needed
kmeans = KMeans(n_clusters=numClusters, random_state=42)
kmeans.fit(X_train)

#Create new collumn in dataframe containing categorization for each house
df['Cluster'] = kmeans.labels_

#Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['LotArea'], df['SalePrice'], c=df['Cluster'], cmap='viridis', alpha=0.6, label='Data Points')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], df['SalePrice'].iloc[:len(centroids)], s=200, c='red', marker='x', label='Centroids')

#Labeling
plt.title('K-Means Clustering on Lot Area')
plt.xlabel('Lot Area (Square Feet)')
plt.ylabel('Sale Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
