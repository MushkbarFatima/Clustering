#!/usr/bin/env python
# coding: utf-8

# In[73]:


#importing libraries
import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


# In[102]:


#Load the data then convert it to dataframe
data1 = arff.loadarff(r"M:\Lab assignment (unsupervised) material-20221130\dataset1_noClusters7.arff")
df1 = pd.DataFrame(data1[0])

#Convert bytes to int
for x in range(len(df1)):
    a=df1.iat[x,2]
    b=int.from_bytes(a,'little')
    df1.iat[x,2]=b

#Load the data then convert it to dataframe
data2 = arff.loadarff(open(r"M:\Lab assignment (unsupervised) material-20221130\dataset2_noClusters3.arff", errors='ignore'))
df2 = pd.DataFrame(data2[0])

#Convert bytes to int
for x in range(len(df2)):
    a=df2.iat[x,2]
    b=int.from_bytes(a,'little')
    df2.iat[x,2]=b

#Load the data then convert it to dataframe 
data3 = arff.loadarff(r"M:\Lab assignment (unsupervised) material-20221130\dataset3_noClusters3.arff")
df3 = pd.DataFrame(data3[0])
#Convert bytes to int
for x in range(len(df3)):
    a=df3.iat[x,2]
    b=int.from_bytes(a,'little')
    df3.iat[x,2]=b

    
data4 = arff.loadarff(r"M:\Lab assignment (unsupervised) material-20221130\dataset4_noClusters2.arff")
df4 = pd.DataFrame(data4[0])

for x in range(len(df4)):
    a=df4.iat[x,2]
    b=int.from_bytes(a,'little')
    df4.iat[x,2]=b

#plotting 4 datasets one by one
fig, ax = plt.subplots(4, figsize=(10, 20))
ax[0].scatter(x = df1['x'], y = df1['y'])
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_title("Dataset 1")

ax[1].scatter(x = df2['x'], y = df2['y'])
ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")
ax[1].set_title("Dataset 2")


ax[2].scatter(x = df3['x'], y = df3['y'])
ax[2].set_xlabel("X")
ax[2].set_ylabel("Y")
ax[2].set_title("Dataset 3")

ax[3].scatter(x = df4['x'], y = df4['y'])
ax[3].set_xlabel("X")
ax[3].set_ylabel("Y")
ax[3].set_title("Dataset 4")
plt.show()


# In[110]:


#USING AVERAGE LINK

#Coping the dataframe to adjust it for algorithm input
df_1 = df1[['x', 'y']].copy()

#Converting the class column into lists
df__1 = df1[['class']].values.flatten()
df11=df__1.tolist()

#for Average link algorithm
gm = AgglomerativeClustering(n_clusters=6).fit(df_1)

#Extracting labels from KMeans
labels1 = gm.labels_
labels=labels1.tolist()
print('Unique Labels:', len(np.unique(labels)))

#Extracting No of clusters
no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

print(f"Adjusted Rand Score: {adjusted_rand_score(df11, labels):.3f}")
print(f"Normalized Mutual Information: {normalized_mutual_info_score(df11, labels):.3f}")

# Generate cluster plot for data
plt.scatter(df1['x'], df1['y'], c=labels, s=50, cmap='viridis')
plt.title('Six clusters of dataset 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[106]:


##Expectation Maximization (EM)

#Coping the dataframe to adjust it for EM input
df_2 = df2[['x', 'y']].copy()

#Converting the class column into lists
df__2 = df2[['class']].values.flatten()
df12=df__2.tolist()

#Gaussian Mixture command
gm = GaussianMixture(n_components=3).fit(df_2)
labels1 = gm.predict(df_2)

#Extracting cluster centres
centers=gm.means_

#Extracting labels from KMeans
labels=labels1.tolist()
print('Unique Labels:', len(np.unique(labels)))

#Extracting No of clusters
no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)
print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

print(f"Adjusted Rand Score: {adjusted_rand_score(df12, labels):.3f}")
print(f"Normalized Mutual Information: {normalized_mutual_info_score(df12, labels):.3f}")

# Generate cluster plot for data
plt.scatter(df2['x'], df2['y'], c=labels, s=50, cmap='viridis')

#Plotting centres on scatter plot
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Three clusters of dataset 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[104]:


#USING KMEANS

#Coping the dataframe to adjust it for KMeans input
df_3 = df3[['x', 'y']].copy()

#Converting the class column into lists
df__3 = df3[['CLASS']].values.flatten()
df13=df__3.tolist()

#KMeans Command
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_3)

#Extracting labels from KMeans
labels1 = kmeans.labels_
labels=labels1.tolist()
print('Unique Labels:', len(np.unique(labels)))

#Extracting No of clusters
no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)
print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

print(f"Adjusted Rand Score: {adjusted_rand_score(df13, labels):.3f}")
print(f"Normalized Mutual Information: {normalized_mutual_info_score(df13, labels):.3f}")

# Generate cluster plot for data
plt.scatter(df3['x'], df3['y'], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_

#Plotting centres on scatter plot
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Three clusters of dataset 3')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[92]:


#USING DBSCAN

#Coping the dataframe to adjust it for DBSCAN input
df_4 = df4[['x', 'y']].copy()

#Converting the class column into lists
df__4 = df4[['class']].values.flatten()
df14=df__4.tolist()

#DBSCAN Command
clustering = DBSCAN(eps=.05, min_samples=3).fit(df_4)

#Extracting labels from DBSAN
labels1=clustering.labels_
labels=labels1.tolist()
print('Unique Labels:', len(np.unique(labels)))

#Extracting No of clusters
no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

print(f"Adjusted Rand Score: {adjusted_rand_score(df14, labels):.3f}")
print(f"Normalized Mutual Information: {normalized_mutual_info_score(df14, labels):.3f}")


# Generate cluster plot for data
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
df4.plot.scatter(x='x', y='y', c=colors, marker="o", picker=True)
plt.title('Two clusters of dataset 4')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

