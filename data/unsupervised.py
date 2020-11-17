import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import decomposition as dc
from sklearn import cluster

def do_PCA(img, variance = .95):
    pca1 = dc.PCA()
    n, d, c = img.shape
    max_dim = 0

    #flatten the image
    flattened_image = np.reshape(img,(n,d*c))

    #PCA for the input image
    pca1.fit(flattened_image)

    #Getting cumulative Variance
    cumm_var = np.cumsum(pca1.explained_variance_ratio_)

    # Check how many eigens explains variance
    k = np.argmax(cumm_var > variance )
    # print("Number of component explaining variance = "+str(k))

    ## reconstruct the image
    PCAF = dc.PCA(n_components=k).fit(flattened_image)

    ## rebuild the compressed image
    Compressed_Image = PCAF.inverse_transform(PCAF.transform(flattened_image))

    ## Change to original colored shape
    Compressed_Image = np.reshape(Compressed_Image, (n,d,c))

    final_cum_variance = PCAF.explained_variance_ratio_

    N = img.shape[0]
    D = img.shape[1]
    if(len(img.shape) == 3):
        denom = N*D*img.shape[2]
        num = k*(1+N+3*D)
    else:
        denom = N*D
        num = k*(1+N+D)

    compression_ratio = num/denom

    return Compressed_Image, compression_ratio, final_cum_variance, k

def cluster_pixels(images, K=50, kmeans=None):
    
    #pass in preprocessed images
    N, H, W, C = images.shape
    flat_images = np.reshape(images, [-1, C]).astype(np.float32)

    if kmeans is None:
        kmeans = cluster.KMeans(n_clusters = K)
        images_new = kmeans.fit_predict(flat_images)
    else:
        images_new = kmeans.predict(flat_images)

    centers = kmeans.cluster_centers_

    new_pixels = centers[images_new]

    reshaped_images = np.reshape(new_pixels, (N, H, W, C))
    
    return kmeans, reshaped_images

#def find_optimal_pallete(images, lower_bound = 50, upper_bound = 200):
    #return an array of color values and the number of color values
    #for each k, try to cluster and see what happens




