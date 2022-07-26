# CS 6375 Assignment 5

from matplotlib import pyplot as io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as image

## "Koala.jpg"
koala = io.imread("Koala.jpg") #image is saved as rows * columns * 3 array
koala_reshape = (koala/255.0).reshape(-1,3)

# k=2
clusters_2 = KMeans(n_clusters=2)
kmeans_2 = clusters_2.fit(koala_reshape)
center_2 = kmeans_2.cluster_centers_
cluster_2 = kmeans_2.labels_
img2 = center_2[cluster_2]
img2 = np.reshape(img2, (koala.shape))
image.imsave('koala2.png',img2)

# k=5
clusters_5 = KMeans(n_clusters=5)
kmeans_5 = clusters_5.fit(koala_reshape)
center_5 = kmeans_5.cluster_centers_
cluster_5 = kmeans_5.labels_
img5 = center_5[cluster_5]
img5 = np.reshape(img5, (koala.shape))
image.imsave('koala5.png',img5)
    
# k=10
clusters_10 = KMeans(n_clusters=10)
kmeans_10 = clusters_10.fit(koala_reshape)
center_10 = kmeans_10.cluster_centers_
cluster_10 = kmeans_10.labels_
img10 = center_10[cluster_10]
img10 = np.reshape(img10, (koala.shape))
image.imsave('koala10.png',img10)

# k=15
clusters_15 = KMeans(n_clusters=15)
kmeans_15 = clusters_15.fit(koala_reshape)
center_15 = kmeans_15.cluster_centers_
cluster_15 = kmeans_15.labels_
img15 = center_15[cluster_15]
img15 = np.reshape(img15, (koala.shape))
image.imsave('koala15.png',img15)

# k=20
clusters_20 = KMeans(n_clusters=20)
kmeans_20 = clusters_20.fit(koala_reshape)
center_20 = kmeans_20.cluster_centers_
cluster_20 = kmeans_20.labels_
img20 = center_20[cluster_20]
img20 = np.reshape(img20, (koala.shape))
image.imsave('koala20.png',img20)

## "Penguins.jpg"
penguin = io.imread("Penguins.jpg")
penguin_reshape = (penguin/255.0).reshape(-1,3)

# k=2
clusters_2 = KMeans(n_clusters=2)
kmeans_2 = clusters_2.fit(penguin_reshape)
center_2 = kmeans_2.cluster_centers_
cluster_2 = kmeans_2.labels_
img2 = center_2[cluster_2]
img2 = np.reshape(img2, (penguin.shape))
image.imsave('penguin2.png',img2)

# k=5
clusters_5 = KMeans(n_clusters=5)
kmeans_5 = clusters_5.fit(penguin_reshape)
center_5 = kmeans_5.cluster_centers_
cluster_5 = kmeans_5.labels_
img5 = center_5[cluster_5]
img5 = np.reshape(img5, (penguin.shape))
image.imsave('penguin5.png',img5)
    
# k=10
clusters_10 = KMeans(n_clusters=10)
kmeans_10 = clusters_10.fit(penguin_reshape)
center_10 = kmeans_10.cluster_centers_
cluster_10 = kmeans_10.labels_
img10 = center_10[cluster_10]
img10 = np.reshape(img10, (penguin.shape))
image.imsave('penguin10.png',img10)

# k=15
clusters_15 = KMeans(n_clusters=15)
kmeans_15 = clusters_15.fit(penguin_reshape)
center_15 = kmeans_15.cluster_centers_
cluster_15 = kmeans_15.labels_
img15 = center_15[cluster_15]
img15 = np.reshape(img15, (penguin.shape))
image.imsave('penguin15.png',img15)

# k=20
clusters_20 = KMeans(n_clusters=20)
kmeans_20 = clusters_20.fit(penguin_reshape)
center_20 = kmeans_20.cluster_centers_
cluster_20 = kmeans_20.labels_
img20 = center_20[cluster_20]
img20 = np.reshape(img20, (penguin.shape))
image.imsave('penguin20.png',img20)