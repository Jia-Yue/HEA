# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 7:34:07 2021

This is the CNN methods coupled with LIME for HEA data

@author: weishi
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(123)

EqHEA_numpy = np.loadtxt("./data/EqHEA_iSFE.txt")
EqHEA = torch.tensor(EqHEA_numpy, dtype=torch.float).reshape(1,-1,40,100)

critical_sigma = torch.rand((1,100))

def load_data():
    raise NotImplementedError()
    
out_size = 1
critical_sigma = torch.rand((1,out_size))

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        #input shape: height = 40, width = 100
        #self.padding = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5),stride=(1,1)) #36x96
        self.conv2 = nn.Conv2d(6, 12, kernel_size=(6,1),stride=(2,1)) #16x96
        self.pool = nn.AvgPool2d((2, 1)) #8x96
        self.conv3 = nn.Conv2d(12, 24, kernel_size=(2,1),stride=(2,1)) #4x96
        self.fc1 = nn.Linear(24*2*96, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 24*2*96)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = convNet()

loss_func = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
#optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.4)

n_epochs = 100

for i in range(n_epochs):
    sigma_hat = network(EqHEA)
    loss = loss_func(sigma_hat, critical_sigma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % (n_epochs // 50) == 0:
        print('{},\t{:.2f}'.format(i, loss.item()))

######### LIME ############

from lime import lime_image

def pred_func(image):
    image_tensor = torch.tensor(image, dtype=torch.float).reshape(1,-1,40,100)
    return network(image_tensor).detach().numpy()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(EqHEA_numpy, 
                                         pred_func, # classification function
                                         top_labels=1, 
                                         hide_color=0, 
                                         num_samples=1)
    
