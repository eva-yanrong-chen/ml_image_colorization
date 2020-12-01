from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data.unsupervised import cluster_classification
from src.util import format_model_input

class ImageDataset(Dataset):

    def __init__(self, data, kmeans=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        features, labels, cluster_label, centers = format_model_input(data, kmeans)

        self.features = features
        self.labels = labels

        self.cluster_label = cluster_label
        self.centers = centers

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.cluster_label[idx], self.labels[idx]