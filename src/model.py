import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50

COLORS = 50

class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.layers = fcn_resnet50(pretrained=True)
      for param in self.parameters():
          param.requires_grad = False

      self.layers.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
      self.layers.classifier[-1] = nn.Conv2d(512, COLORS, kernel_size=(1, 1), stride=(1, 1))
      self.layers.aux_classifier[-1] = nn.Conv2d(256, COLORS, kernel_size=(1, 1), stride=(1, 1))
      self.loss_criterion = nn.MSELoss()
      
    def forward(self, x):
      return self.layers(x)
    