from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

class MNIST_train_subset(Dataset):
    def __init__(self, data, labels, mainset_idx):
        self.data = data
        self.labels = labels
        self.mainset_idx = mainset_idx
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.mainset_idx[index] # idx allow keep track of the loss value of each sample

class MNIST_train(Dataset):
    def __init__(self, numGroups=5, valSplit=0):
        # Load the MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.valSplit = valSplit
        self.data = self.dataset.data
        self.labels = self.dataset.targets
        self.numGroups = numGroups
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
            valSet = MNIST_train_subset(valData, valLabels, valIndices)
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
            subset = MNIST_train_subset(group_data, group_labels, group_indices)
            self.groups.append(subset)
            self.corrIdx[i] = group_indices

# testing
if __name__ == "__main__":
    dataset = MNIST_train(1)
    loader = [DataLoader(i, batch_size=32, shuffle=True) for i in dataset.groups]
    for i in range(3):
        for batch in loader[0]:
            data, labels, idx = batch
            print(data.shape, labels.shape, idx.shape)