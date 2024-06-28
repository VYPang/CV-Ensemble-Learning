import torch.nn as nn
import torch.nn.functional as f

class softmax_CNN(nn.Module):
    def __init__(self, numClass):
        super().__init__()
        self.numClass = numClass

        # convolution layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )

        # fully connected layer
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return nn.functional.log_softmax(x, dim=1)
