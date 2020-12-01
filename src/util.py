import numpy as np
import torch
import cv2

from data.unsupervised import cluster_classification

# Takes data from dataset
def format_model_input(data, kmeans):

    # Retrieve ab channels
    labels = data[:, :, :, 1:]

    # Get clustering image from ab channels
    print("Starting conversion to classification classes")
    centers, cluster_label = cluster_classification(labels, kmeans)
    print("Finished conversion to classification classes")

    # Transpose data for opencv input format
    data = np.transpose(data, (0, 3, 1, 2))
    data = torch.tensor(data)

    # Transpose cluster labeling for torch input
    cluster_label = torch.LongTensor(cluster_label)

    # Normalize input
    data_float = data / 255

    # Split the X's and labels
    input = data_float[:, 0:1, :, :]
    label = data_float[:, 1:, :, :]

    return input, label, cluster_label, centers

def show_cv2_image(image):

    image_int = (image * 255)
    image = np.uint8(image_int.detach())
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)

# Converts Singular torch image back to ab space
def convert_to_ab(image, centers):
    # pass in preprocessed images
    C, H, W = image.shape

    images_new = torch.argmax(image, dim=0)

    flat_images = torch.flatten(images_new)

    # flat_images = (flat_images + 5) % 50

    new_pixels = torch.tensor(centers[flat_images]).T

    reduced_image = torch.reshape(new_pixels, (-1, H, W))

    return reduced_image