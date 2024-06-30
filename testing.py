import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.model import AlexNet
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pickle
from data.dataLoading import testSource
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def testing(model, lossFunction, config, testLoader):
    total_loss = 0
    accurates = 0
    model.eval()
    lossRecord = {i:[] for i in range(10)}
    accRecord = [0 for _ in range(10)]
    countRecord = [0 for _ in range(10)]
    vectorRecord = {}
    test_tqdm = tqdm(testLoader, total=len(testLoader))
    for batch_idx, batch in enumerate(test_tqdm):
        test_tqdm.set_description(f'Testing')

        x, y, mainSet_idx = batch
        y = y.to(device)
        x = x.float().to(device)
        if len(x.shape) == 3:
            x = x[:, None, ...]
        output, vector = model(x)
        vectorRecord[mainSet_idx.cpu().numpy()[0]] = [y.item(), vector.cpu().detach().numpy()[0]]

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
    return lossRecord, accRecord, countRecord, vectorRecord

def graphPerf(lossRecord, accRecord, countRecord, config):
    classes = config.data.classes
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
        ax.set_title(f'Class {classes[i]}')
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
    ax.set_xticklabels([f'{classes[i]}' for i in range(len(accuracy))])
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
        print(f'{classes[i]} \t(total counts: {countRecord[i]}\taccuracy: {round(accuracy[i], 4)})')

def pcaAnalysis(vectorRecord, config):
    classes = config.data.classes
    vectors = np.array([vectorRecord[i][1] for i in range(len(vectorRecord))])
    pca = PCA(n_components=3)
    print('fitting pca...')
    pca.fit(vectors)

    # group vectors by class
    vectorGroup = {i:[] for i in range(10)}
    mainSetIdx = {i:[] for i in range(10)}
    for i in range(len(vectorRecord)):
        vectorGroup[vectorRecord[i][0]].append(vectorRecord[i][1])
        mainSetIdx[vectorRecord[i][0]].append(i)
    
    # plot pca
    fig = go.Figure()
    for i in range(len(vectorGroup)):
        data = pca.transform(np.array(vectorGroup[i]))
        text = [f'mainSetIdx: {idx}' for idx in mainSetIdx[i]]
        fig.add_trace(go.Scatter3d(text=text,
                                    x=data[:, 0],
                                    y=data[:, 1],
                                    z=data[:, 2],
                                    mode='markers',
                                    marker=dict(size=2.5),
                                    name=f'class {classes[i]}'
                                   ))
    fig.update_layout(scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3'
    ))
    fig.write_html('pca.html')
    print('pca analysis done!')

if __name__ == "__main__":
    configPath = 'configuration/config.yaml'
    modelPath = 'ckpt/whole/final.pt'
    config = OmegaConf.load(configPath)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    numClass = len(config.data.classes)
    channel = config.data.shape[0]
    model = AlexNet(channel, numClass, test=True).to(device)
    lossFunction = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(modelPath, map_location=device))

    # load dataset
    testSet = testSource()
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False)

    # test
    lossRecord, accRecord, countRecord, vectorRecord = testing(model, lossFunction, config, testLoader)
    graphPerf(lossRecord, accRecord, countRecord, config)
    pcaAnalysis(vectorRecord, config)