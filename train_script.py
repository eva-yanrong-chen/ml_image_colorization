import torch
import cv2
import numpy as np
from src.model import Model
from data.data_loading import retrieve_data

from src.util import format_model_input, show_cv2_image

if __name__  =='__main__':

    train, val, test = retrieve_data("bird")

    m_input, _ = format_model_input(train)
    model = Model()
    print("Starting model")
    y = model(m_input[:3])
    print("Finished model")
    # print(y['out'][0].shape, y['aux'][0].shape)

    x1 = torch.cat((m_input[0], y['out'][0]), axis=0)

    show_cv2_image(x1)