from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self,rho = 0.001):
        super(ConvNet, self).__init__()
        #self.inNorm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, 7, 1, padding=3)
        #self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.fc1 = nn.Linear(4*4*64, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)