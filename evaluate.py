import torch
import cv2
import numpy as np
from src.model import Model
from data.data_loading import retrieve_data
from torch.utils.data import Dataset, DataLoader

from src.util import format_model_input, show_cv2_image
from data.dataloader import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_images(annotation):
    train, val, test = retrieve_data(annotation)

    m_input, labels = format_model_input(val)

    model = Model()
    model.load_state_dict(torch.load('weights/{}.pt'.format(annotation)))
    model.eval()

    while True:
        i = np.random.randint(len(m_input))

        x = m_input[i:i+1]

        y_t = labels[i:i+1]
        y = model(x)

        loss = model.loss_criterion(y['out'][0], y_t[0])
        created = torch.cat((x[0], y['out'][0]), axis=0)
        ground_t = torch.cat((x[0], y_t[0]), axis=0)

        print("Loss: {}".format(loss))

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