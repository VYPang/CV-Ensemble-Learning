import torch.nn as nn
import torch.nn.functional as f

class basicCNN(nn.Module):
    def __init__(self, numClass, test=False):
        super().__init__()
        self.numClass = numClass
        self.test = test

        # convolution layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),
        )

        # fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.output = nn.Linear(100, numClass)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        if self.test:
            vector = x
        x = self.output(x)
        if self.test:
            return f.log_softmax(x, dim=1), vector
        return f.log_softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self, channel, num_class, test=False):
        super(AlexNet, self).__init__()
        self.test = test
        self.features = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        if self.test:
            vector = x
        x = self.output(x)
        if self.test:
            return f.log_softmax(x, dim=1), vector
        return f.log_softmax(x, dim=1)