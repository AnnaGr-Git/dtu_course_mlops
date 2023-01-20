import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset

def random_data():
    # exchange with the corrupted mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 
    return train, test

def get_mnist_data(folderpath, train: bool = True):
    filepaths = glob.glob(os.path.join(folderpath, "*.npz")) 

    if train:
        search_word = "train"
    else:
        search_word = "test"

    mnist_data = {}
    for file in filepaths:
        data = np.load(file)
        images = data['images']
        labels = data['labels']

        if search_word in file:
            if len(mnist_data) == 0:
                mnist_data['images'] = images
                mnist_data['labels'] = labels
            else:
                mnist_data['images'] = np.concatenate((mnist_data['images'], images), axis=0)
                mnist_data['labels'] = np.concatenate((mnist_data['labels'], labels), axis=0)

    return mnist_data['images'], mnist_data['labels']

class MnistDataset(Dataset):
    def __init__(self, dataset_dir, train, transform=None, target_transform=None):
        self.dataset_dir = dataset_dir
        self.images, self.labels = get_mnist_data(dataset_dir, train)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
