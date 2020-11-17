from data.data_loading import retrieve_data
from data.unsupervised import cluster_pixels
import cv2
import numpy as np
from joblib import dump, load

def segment_by_annotation(annotation):
    train, val, test = retrieve_data(annotation)

    images = train[0:2]

    image = cv2.cvtColor(images[1], cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)

    try:
        kmeans = load('weights/{}_kmeans.joblib'.format(annotation))
        print("Loaded Previous parameters")
        _, pic_changed = cluster_pixels(images,kmeans=kmeans)
    except:
        print("Could not find old Parameters")
        kmeans, pic_changed = cluster_pixels(images, K=50)

        dump(kmeans, 'weights/{}_kmeans.joblib'.format(annotation))
    pic_int = np.uint8(pic_changed)

    image = cv2.cvtColor(pic_int[1], cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)



if __name__  =='__main__':
    segment_by_annotation('bird')