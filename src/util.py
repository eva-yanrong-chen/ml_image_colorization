import numpy as np
import torch
import cv2

# Takes data from dataset
def format_model_input(data):

    # Transpose data for opencv input format
    data = np.transpose(data, (0, 3, 1, 2))
    data = torch.tensor(data)

    # Normalize input
    data_float = data / 255

    # Split the X's and labels
    input = data_float[:, 0:1, :, :]
    label = data_float[:, 1:, :, :]

    return input, label

def show_cv2_image(image):

    image_int = (image * 255)
    image = np.uint8(image_int.detach())
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imshow("", image)
    cv2.waitKey(0)