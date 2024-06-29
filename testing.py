import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.model import softmax_CNN
import numpy as np
from omegaconf import OmegaConf
from torchvision import datasets, transforms

def testing(model, lossFunction, config, testLoader):
    total_loss = 0
    accurates = 0
    model.eval()
    test_tqdm = tqdm(testLoader, total=len(testLoader))
    for batch_idx, batch in enumerate(test_tqdm):
        test_tqdm.set_description(f'Testing')

        x, y = batch
        y = y.to(device)
        x = x.to(device)
        output = model(x)

        # calculate loss
        loss = lossFunction(output, y)
        total_loss += loss

        # calculate accuracy
        _, predicted = torch.max(output, 1)
        accurates += (predicted == y).sum().item()
    print(f'Loss: {total_loss/len(testLoader)}')
    print(f'Accuracy: {accurates/len(testLoader)}')

if __name__ == "__main__":
    configPath = 'configuration/config.yaml'
    modelPath = 'ckpt/A/final.pt'
    numGroups = 5   # number of groups splited in training set
    config = OmegaConf.load(configPath)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    numClass = 10
    model = softmax_CNN(numClass).to(device)
    lossFunction = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(modelPath))

    # load dataset
    testSet = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False)

    # train
    testing(model, lossFunction, config, testLoader)