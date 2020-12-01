import torch
import cv2
import numpy as np
from src.model import Model
from data.data_loading import retrieve_data
from torch.utils.data import Dataset, DataLoader

from src.util import format_model_input, show_cv2_image, convert_to_ab
from data.dataloader import ImageDataset
from joblib import load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, val_loader, num, centers):
    for n in range(num):

        for i, data in enumerate(val_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, cluster_labels, labels = data

            # forward
            outputs = model(inputs)

            loss = model.loss_criterion(outputs['out'], cluster_labels)

            # Get rid of batch dimension
            input = inputs[0]
            output = outputs['out'][0]
            label = labels[0]

            ab_output = convert_to_ab(output, centers)

            created = torch.cat((input, ab_output), axis=0)
            ground_t = torch.cat((input, label), axis=0)

            print("Loss: {}".format(loss))
            show_cv2_image(ground_t)
            show_cv2_image(created)


def evaluate_images(annotation):
    train, val, test = retrieve_data(annotation)

    kmeans = load('weights/{}_kmeans.joblib'.format(annotation))

    model = Model()
    model.load_state_dict(torch.load('weights/{}.pt'.format(annotation)))
    model.eval()

    val_dataset = ImageDataset(val, kmeans)

    data_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, num_workers=0)
    eval_model(model, data_loader, 10, val_dataset.centers)


if __name__  =='__main__':
    evaluate_images('plant_life')

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