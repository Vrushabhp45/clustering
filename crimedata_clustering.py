# Importing Relevant Library
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
# Loading the data
data = pd.read_csv("C:/Users/HP/PycharmProjects/Excelrdatascience/crime_data.csv")
data
data.info()
data2=data.drop(['Unnamed: 0'],axis=1)
data2
# Selecting the feature
from sklearn.preprocessing import normalize
data_scaled = normalize(data2)
data_scaled = pd.DataFrame(data_scaled, columns=data2.columns)
data_scaled.head()
# To find No of Cluster use WCSS and Elbow Method
wcss=[]
for i in range(1,7):
  kmeans = KMeans(i)
  kmeans.fit(data_scaled)
  wcss_iter = kmeans.inertia_
  wcss.append(wcss_iter)

number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
# we can choose 3 as no. of clusters, this method shows what is the good number of clusters.
# Let See Dendogram to find the no of cluster
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
# Here also with the help of dendogram we see 3 cluster
# 1.K-Means
from sklearn.cluster import KMeans
k_means=KMeans(n_clusters=4,random_state=42)
k_means.fit(data_scaled)
identified_clusters = kmeans.fit_predict(data_scaled)
identified_clusters
y = pd.DataFrame(data=identified_clusters, columns=['clusters'])
y
# Adding clusters to dataset
data_scaled['clusters']=identified_clusters
data_scaled
# Plotting resulting clusters
data_scaled['KMeans_labels']=k_means.labels_

colors=['purple','red','blue','green']
plt.figure(figsize=(10,10))
plt.scatter(data_scaled['Murder'],data_scaled['clusters'],c=data_scaled['KMeans_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('K-Means Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()
# 2. Hierachial Clustering
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=4, affinity='euclidean')
model.fit(data_scaled)
# Plotting resulting clusters
data_scaled['HR_labels']=model.labels_

plt.figure(figsize=(10,10))
plt.scatter(data_scaled['Murder'],data_scaled['clusters'],c=data_scaled['HR_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('Hierarchical Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()
# 3.DBSCAM CLUSTERING
from sklearn.cluster import DBSCAN
dbscan=DBSCAN()
dbscan.fit(data_scaled)
# Plotting resulting clusters
data_scaled['DBSCAN_labels']=dbscan.labels_

plt.figure(figsize=(10,10))
plt.scatter(data_scaled['Murder'],data_scaled['clusters'],c=data_scaled['DBSCAN_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAM CLUSTERING',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()