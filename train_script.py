import torch
import cv2
import numpy as np
from src.model import Model
from data.data_loading import retrieve_data
from torch.utils.data import Dataset, DataLoader
from joblib import load

from src.util import format_model_input, show_cv2_image
from data.dataloader import ImageDataset

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(trainloader, optimizer, net, criterion, epochs=NUM_EPOCHS):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs['out'], labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # if i % 20 == 19:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
            # print("Finished iteration {}".format(i))
    print('Finished Training')


def train_by_annotation(annotation):
    train, val, test = retrieve_data(annotation)

    model = Model()
    model.to(device)

    # Get kmeans clustering for particular annotation
    kmeans = load('weights/{}_kmeans.joblib'.format(annotation))
    train_dataset = ImageDataset(train, kmeans)

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(dataloader, optimizer, model, criterion)

    torch.save(model.state_dict(), "weights/{}.pt".format(annotation))

if __name__  =='__main__':
    train_by_annotation('plant_life')

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