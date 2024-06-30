from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np
import random

class testSource(Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #self.dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        self.dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.data = self.dataset.data
        self.labels = self.dataset.targets
        self.mainset_idx = list(range(len(self.data)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label, mainset_idx = self.data[index], self.labels[index], self.mainset_idx[index]
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).float()
        label = torch.tensor(label)
        return data, label, mainset_idx

class trainSubset(Dataset):
    def __init__(self, data, labels, mainset_idx, augmentation=False):
        self.data = data
        self.labels = labels
        self.mainset_idx = mainset_idx
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label, mainset_idx = self.data[index], self.labels[index], self.mainset_idx[index]
        if self.augmentation:
            height, width, channel = data.shape
            data = data.copy()
            if np.random.rand() < 0.5:
                data = np.flip(data, axis=1)
            # shift the colored image
            if np.random.rand() < 0.5:
                dx, dy = np.random.randint(-2, 3, 2)
                data =  cv2.warpAffine(data, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))
            # rotate the image
            if np.random.rand() < 0.5:
                angle = np.random.randint(-15, 16)
                M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                data = cv2.warpAffine(data, M, (width, height))
            # change the brightness
            data = cv2.convertScaleAbs(data, 
                                       alpha=np.random.uniform(0.9, 1.1),
                                       beta=np.random.randint(-10, 11))
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).float()
        return data, label, mainset_idx # idx allow keep track of the loss value of each sample

class trainSource(Dataset):
    def __init__(self, numGroups=5, valSplit=0, augmentation=False):
        # Load the MNIST dataset
        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        #self.dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        self.dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.valSplit = valSplit
        self.data = self.dataset.data
        self.labels = self.dataset.targets
        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels)
        self.numGroups = numGroups
        self.augmentation = augmentation
        self.create_groups()

    # dataset.groups[0] is the first group containing [data, labels]
    def create_groups(self):
        self.groups = []
        self.corrIdx = {}
        allIdx = list(range(len(self.data)))
        random.Random(0).shuffle(allIdx) # shuffle the indices

        if self.valSplit > 0:
            valSize = int(len(self.data) * self.valSplit)
            valIndices = allIdx[:valSize]
            valData = self.data[valIndices]
            valLabels = self.labels[valIndices]
            valSet = trainSubset(valData, valLabels, valIndices)
            self.valSet = valSet
            self.corrIdx['val'] = valIndices
            allIdx = allIdx[valSize:]

        groupSize = len(allIdx) // self.numGroups
        for i in range(self.numGroups):
            start = i * groupSize
            end = (i + 1) * groupSize
            # check last group and end != len(data)
            if i == self.numGroups - 1 and end != len(allIdx):
                end = len(self.data)
            group_indices = allIdx[start:end]
            group_data = self.data[group_indices]
            group_labels = self.labels[group_indices]
            subset = trainSubset(group_data, group_labels, group_indices, augmentation=self.augmentation)
            self.groups.append(subset)
            self.corrIdx[i] = group_indices

# testing
if __name__ == "__main__":
    dataset = trainSource(1, valSplit=0.1, augmentation=True)
    loader = [DataLoader(i, batch_size=1, shuffle=True) for i in dataset.groups]
    val = DataLoader(dataset.valSet, batch_size=1, shuffle=True)
    # for i in range(3):
    #     for batch in loader[0]:
    #         data, labels, idx = batch
    #         print(data.shape, labels.shape, idx.shape)

    for batch in val:
        data, labels, idx = batch
        print(data.shape, labels.shape, idx.shape)
        break