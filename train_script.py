import torch
import cv2
import numpy as np
from src.model import Model
from data.data_loading import retrieve_data

train, val, test = retrieve_data("bird")
train = np.transpose(train, (0, 3, 1, 2))
train = torch.tensor(train)
train_float = train / 255
model = Model()
y = model(train_float[:3, 0:1, :, :])
print(y['out'][0].shape, y['aux'][0].shape)

x1 = torch.cat((train_float[0, 0:1, :, :], y['out'][0]), axis=0)
image = (x1 * 255)
image = np.uint8(image.detach())
image = np.transpose(image, (1, 2, 0))

print(image.dtype)

image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
cv2.imshow("", image)
cv2.waitKey(0)