import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn import preprocessing

def OurBagVW(data_train, data_test):
    k = 600
    kmeans = KMeans(k) # n_clusters = k 
    # concatenate all features into a global array of features
    all_features = np.concatenate(data_train)
    # create a subset about 50% of all features
    # first we take randomly indeces from the set
    indeces = np.random.randint(len(all_features), size=len(all_features)//2) 
    less_features = all_features[indeces]
    # fit the kmean => calculate the centroids
    kmeans.fit(less_features)
    centroids = kmeans.cluster_centers_
    edges = np.arange(1,k+2)
    BOF_tr = np.empty((len(data_train),k))
    indexs= np.array([[]])
    for i in range(len(data_train)):
        dist = distance.cdist(data_train[i], centroids, 'euclidean')
        indexs = np.argmin(dist,axis = 1)
        hist,_= np.histogram(indexs,edges)
        BOF_tr[i] = preprocessing.normalize(hist.reshape((len(hist),1)), norm='l2').squeeze()
       
       
    BOF_ts = np.empty((len(data_test),k))
    for i in range(len(data_test)):
        dist = distance.cdist(data_test[i], centroids, 'euclidean')
        indexs = np.argmin(dist,axis = 1)
        hist,_= np.histogram(indexs,edges)
        BOF_ts[i] = preprocessing.normalize(hist.reshape((len(hist),1)), norm='l2').squeeze()
     
    return BOF_tr, BOF_ts