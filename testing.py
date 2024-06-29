import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.model import softmax_CNN
import numpy as np
from omegaconf import OmegaConf
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pickle

def testing(model, lossFunction, config, testLoader):
    total_loss = 0
    accurates = 0
    model.eval()
    lossRecord = {i:[] for i in range(10)}
    accRecord = [0 for i in range(10)]
    countRecord = [0 for i in range(10)]
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

        # record performance
        lossRecord[y.cpu().numpy()[0]].append(loss.item())
        if predicted == y:
            accRecord[y.item()] += 1
        countRecord[y.item()] += 1
        
    print(f'Over All Loss: {total_loss/len(testLoader)}')
    print(f'Over All Accuracy: {accurates/len(testLoader)}')
    return lossRecord, accRecord, countRecord

def graphPerf(lossRecord, accRecord, countRecord):
    # plot loss histogram
    all_losses = [loss for losses in lossRecord.values() for loss in losses]
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
    # Iterate over the keys (classes) in the dictionary
    for i, (ax, losses) in enumerate(zip(axes.flat, lossRecord.values())):
        # Plot the histogram for the current class
        ax.hist(losses, bins=50, range=(min_loss, max_loss))
        ax.set_title(f'Class {i}')
        ax.set_xlabel('Loss Value')
        ax.set_ylabel('Frequency')
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    # Show the plot
    plt.savefig('lossHistogram.jpg')
    plt.clf()

    # plot accuracy histogram
    accuracy = [accRecord[i]/countRecord[i] for i in range(len(accRecord))]
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the bar chart
    ax.bar(range(len(accuracy)), accuracy)
    # Set the x-axis ticks and labels
    ax.set_xticks(range(len(accuracy)))
    ax.set_xticklabels([f'{i}' for i in range(len(accuracy))])
    # Set the axis labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Class')
    # Adjust the layout
    plt.tight_layout()
    # Show the plot
    plt.savefig('accuracyBar.jpg')

    # info printing
    for i in range(len(accRecord)):
        print(f'Number {i} (total counts: {countRecord[i]}\taccuracy: {round(accuracy[i], 4)})')


if __name__ == "__main__":
    configPath = 'configuration/config.yaml'
    modelPath = 'ckpt/whole/final.pt'
    numGroups = 5   # number of groups splited in training set
    config = OmegaConf.load(configPath)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    numClass = 10
    model = softmax_CNN(numClass).to(device)
    lossFunction = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(modelPath, map_location=device))

    # load dataset
    testSet = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False)

    # test
    lossRecord, accRecord, countRecord = testing(model, lossFunction, config, testLoader)
    graphPerf(lossRecord, accRecord, countRecord)
