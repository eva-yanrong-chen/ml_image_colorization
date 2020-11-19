from data.data_loading import retrieve_data
from data.unsupervised import cluster_pixels
import cv2
import numpy as np
from joblib import dump, load

def segment_by_annotation(annotation):
    train, val, test = retrieve_data(annotation)

    # Shape:
    # N x H x W x C
    images = train[:20]


    # Segment out light layer:
    l = images[:, :, :, :1]
    ab = images[:, :, :, 1:]

    image = cv2.cvtColor(images[0], cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)

    try:
        kmeans = load('weights/{}_kmeans.joblib'.format(annotation))
        print("Loaded Previous parameters")
        _, pic_changed = cluster_pixels(images, kmeans=kmeans)
    except:
        print("Could not find old Parameters")
        kmeans, pic_changed = cluster_pixels(ab, K=20)


        dump(kmeans, 'weights/{}_kmeans.joblib'.format(annotation))
    print("Finished clustering")
    pic_int = np.uint8(pic_changed)

    pic = np.concatenate((l, pic_int), axis=-1)

    image = cv2.cvtColor(pic[0], cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)

if __name__  =='__main__':
    segment_by_annotation('plant_life')