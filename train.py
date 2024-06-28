import torch
import torch.nn as nn
from tqdm import tqdm
from data.MNIST.dataLoading import MNIST_train
from torch.utils.data import DataLoader
from utils.model import softmax_CNN
import torch.optim as optim
import numpy as np
import datetime
from omegaconf import OmegaConf
import os

def train(savePath, model, optimizer, lossFunction, config, trainLoader, valLoader=None):
    epochs = config.train.epochs
    for epoch in range(epochs):
        # training loop
        total_loss = 0
        train_tqdm = tqdm(trainLoader, total=len(trainLoader))
        for batch_idx, batch in enumerate(train_tqdm):
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs}')
            model.train()
            optimizer.zero_grad()

            x, y, _ = batch
            y = y.to(device)
            x = x.float().to(device)[:, None, ...]

            output = model(x)
            loss = lossFunction(output, y)
            total_loss += loss
            loss.backward()
            optimizer.step()
            average_loss = round((total_loss.detach().cpu().numpy()/ (batch_idx + 1)), 5)
            train_tqdm.set_postfix(loss=average_loss)

        # validation loop
        if valLoader != None:
            total_val_loss = 0
            val_tqdm = tqdm(valLoader, total=len(valLoader))
            for batch_idx, batch in enumerate(val_tqdm):
                val_tqdm.set_description('Validation')
                model.eval()
                x, y, _ = batch
                y = y.to(device)
                x = x.float().to(device)[:, None, ...]

                output = model(x)
                loss = lossFunction(output, y)
                total_val_loss += loss
                average_val_loss = round((total_val_loss.detach().cpu().numpy()/ (batch_idx + 1)), 5)
                val_tqdm.set_postfix(val_loss=average_val_loss)
        print('\n')
        
        if epoch+1 % 100 == 0:
            torch.save(model, savePath + f'/epoch{epoch+1}-{average_loss}-{average_val_loss}.pt')
    torch.save(model, savePath + f'/final-{average_loss}-{average_val_loss}.pt')
    print('Model saved, all done!')

if __name__ == "__main__":
    configPath = 'configuration/config.yaml'
    numGroups = 1   # number of groups splited in training set
    config = OmegaConf.load(configPath)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    numClass = 10
    model = softmax_CNN(numClass).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.train.lr, momentum=config.train.momentum)
    lossFunction = nn.CrossEntropyLoss()

    # create checkpoint folder
    current_time = datetime.datetime.now()
    file_name = current_time.strftime("%Y-%m-%d_%H%M%S")
    savePath = f'ckpt/{file_name}'
    os.makedirs(savePath)

    # load dataset
    trainSet = MNIST_train(numGroups=numGroups, valSplit=config.train.val_split)
    trainLoader = [DataLoader(i, batch_size=config.train.batch_size, shuffle=True) for i in trainSet.groups]
    if config.train.val_split > 0:
        valLoader = DataLoader(trainSet.valSet, batch_size=config.train.batch_size, shuffle=True)
    else:
        valLoader = None

    # train
    for i in range(len(trainLoader)):
        train(savePath, model, optimizer, lossFunction, config, trainLoader[i], valLoader)