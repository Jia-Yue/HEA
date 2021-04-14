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
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np
import os

torch.manual_seed(666)

glide = 3
depth = 2
core = 20
num_elem = 5

def feature_scaling(composition):
    if composition.shape[1] != core*depth*glide:
        raise ValueError("shape of input data is not compatible with \
                         core*depth*glide")
        
    scalers = {}
    for i in range(composition.shape[2]):
        scalers[i] = StandardScaler()
        composition[:, i, :] = scalers[i].fit_transform(composition[:, i, :]) 
        
    return composition

def mirror(comp, sigma, axis):
    if axis == "glide":
        raise NotImplementedError()
    elif axis == "core":
        mirror_comp = np.flip(comp, axis=1)
        
        return np.concatenate((comp,mirror_comp), axis=0), \
            np.concatenate((sigma, sigma))
            
def periodic_augmentation(comp, sigma):
    comp_aug = comp
    sigma_aug = sigma
    rolled_comp = comp
    for i in range(comp.shape[1]-1):
        rolled_comp = np.roll(rolled_comp, 1, axis=1)
        comp_aug = np.concatenate((comp_aug, rolled_comp),axis=0)
        sigma_aug = np.concatenate((sigma_aug, sigma),axis=0)
    
    return comp_aug.reshape(-1, core, glide, depth, num_elem).swapaxes(1,4), \
        sigma_aug
        
def load_data(folder_path = "./data", scaling=True):
    PATH = folder_path
    comp = np.load(os.path.join(PATH, "Composition.npy"))
    sigma = np.load(os.path.join(PATH, "CStress.npy"))
    
    X_train, X_test, Y_train, Y_test = train_test_split(comp, sigma, \
                                                        test_size = 0.1)
    
    if scaling is True:
        X_train = feature_scaling(X_train)
    X_train = X_train.reshape(-1,core,glide*depth, num_elem)
    X_test = X_test.reshape(-1,core,glide, depth, num_elem).swapaxes(1,4)

    X_train, Y_train = mirror(X_train, Y_train, axis="core")
    
    X_train, Y_train = periodic_augmentation(X_train, Y_train)
    
    x_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(Y_train).float()
    x_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(Y_test).float()
    
    train = TensorDataset(x_train,y_train)
    test = TensorDataset(x_test,y_test)
    
    # data loader
    batch_size = 500
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = 1, shuffle = False)
    
    return train_loader, test_loader
    
# shape = (batch_size, channel, glide, depth, core)
train_loader, test_loader = load_data()
batch_size = 500
out_size = 1

class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        #input shape: batch_size = 100, channel = 5, glide = 3, 
        # depth = 2, core = 20
        #self.padding = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv3d(5, 32, kernel_size=(2,2,5)) #2x1x16
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2,1,4)) #1x1x13
        #self.pool = nn.AvgPool2d((2, 1)) #8x96
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(1,1,3)) #1x1x11
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(1,1,3)) #1x1x9
        self.fc1 = nn.Linear(9*256, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, out_size)
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = convNet()

loss_func = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
#optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.4)

n_epochs = 4

count = 0
train_log = []
test_log = []
count_log = []
for i in range(n_epochs):
    for features, labels in train_loader:
        features_train = Variable(features.view(-1,num_elem,glide,depth,core))
        labels_train = Variable(labels)
        sigma_hat = network(features_train)
        loss = loss_func(sigma_hat, labels_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
        if count % 10 == 0:
            train_log.append(loss.item())
            count_log.append(count)
            loss_test = 0
            for testfeatures, testlabels in test_loader:
                features_test = Variable(testfeatures.view(-1,num_elem,glide, \
                                                           depth,core))
                labels_test = Variable(testlabels)
                sigma_hat_test = network(features_test)
                loss_test_iter = loss_func(sigma_hat_test, labels_test)
                loss_test += loss_test_iter.item()
            test_log.append(loss_test/len(test_loader))
            
        if count % 50 == 0:
            print('Iteration: {}  Train Loss: {}  Test Loss:{} \
                  %'.format(count, train_log[-1], test_log[-1]))
            
plt.plot(count_log, train_log, color='green')
plt.plot(count_log, test_log, color='red')
                
plt.plot(labels_train.detach().numpy(), color='green')     
plt.plot(network(features_train).detach().numpy(), color='red')     

######### LIME ############

# =============================================================================
# from lime import lime_image
# 
# def pred_func(image):
#     image_tensor = torch.tensor(image, dtype=torch.float).reshape(1,-1,40,100)
#     return network(image_tensor).detach().numpy()
# 
# explainer = lime_image.LimeImageExplainer()
# explanation = explainer.explain_instance(EqHEA_numpy, 
#                                          pred_func, # classification function
#                                          top_labels=1, 
#                                          hide_color=0, 
#                                          num_samples=1)
# =============================================================================
    
