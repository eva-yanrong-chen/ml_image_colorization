import torch
import cv2
import numpy as np
from src.model import Model
from data.data_loading import retrieve_data
from torch.utils.data import Dataset, DataLoader

from src.util import format_model_input, show_cv2_image
from data.dataloader import ImageDataset


def evaluate_images(annotation):
    train, val, test = retrieve_data(annotation)

    m_input, labels = format_model_input(val)

    model = Model()
    model.load('weights/{}.pt'.format(annotation))

    while True:
        i = np.random.randint(len(m_input))

        x = m_input[i:i+1]

        y_t = labels[i:i+1]
        y = model(x)

        created = torch.cat((x[0], y['out'][0]), axis=0)
        ground_t = torch.cat((x[0], y_t[0]), axis=0)

        show_cv2_image(ground_t)
        show_cv2_image(created)

if __name__  =='__main__':
    evaluate_images('bird')

    # train, val, test = retrieve_data("bird")
    #
    # m_input, _ = format_model_input(train)
    # model = Model()
    # print("Starting model")
    # y = model(m_input[:3])
    # print("Finished model")
    # # print(y['out'][0].shape, y['aux'][0].shape)
    #
    # x1 = torch.cat((m_input[0], y['out'][0]), axis=0)
    #
    # show_cv2_image(x1)