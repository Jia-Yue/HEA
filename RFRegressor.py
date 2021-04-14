# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:50:44 2021

@author: weishi
"""
import numpy as np
import os

import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
    return X_train.reshape(-1,num_elem*glide*depth*core), \
        X_test.reshape(-1,num_elem*glide*depth*core), Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_data()

xgbr = xgb.XGBRegressor(verbosity=0) 
xgbr.fit(X_train, Y_train)

score = xgbr.score(X_train, Y_train)  
print("Training score: ", score)

Y_pred = xgbr.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("MSE: %.2f" % mse)

x_ax = range(len(Y_test))
plt.plot(x_ax, Y_test, label="original")
plt.plot(x_ax, Y_pred, label="predicted")
plt.title("Critical stress prediction")
plt.legend()
plt.show()