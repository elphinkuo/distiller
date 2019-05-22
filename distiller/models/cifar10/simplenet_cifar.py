#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['simplenet_cifar']


class Simplenet(nn.Module):
#    def __init__(self):
#        super(Simplenet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.relu_conv1 = nn.ReLU()
#        self.pool1 = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.relu_conv2 = nn.ReLU()
#        self.pool2 = nn.MaxPool2d(2, 2)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.relu_fc1 = nn.ReLU()
#        self.fc2 = nn.Linear(120, 84)
#        self.relu_fc2 = nn.ReLU()
#        self.fc3 = nn.Linear(84, 10)

#    def forward(self, x):
#        x = self.pool1(self.relu_conv1(self.conv1(x)))
#        x = self.pool2(self.relu_conv2(self.conv2(x)))
#        x = x.view(-1, 16 * 5 * 5)
#        x = self.relu_fc1(self.fc1(x))
#        x = self.relu_fc2(self.fc2(x))
#        x = self.fc3(x)
#        return x
    def __init__(self,rho = 0.001):
        super(Simplenet, self).__init__()
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




def simplenet_cifar():
    model = Simplenet()
    return model
