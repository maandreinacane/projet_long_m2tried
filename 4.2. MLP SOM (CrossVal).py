#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP SOM (CrossVal)
"""

import library as lib

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error

#----------------------------------------------------------------------
# Clear the Screen
#----------------------------------------------------------------------
cls = lambda: print('\n'*50)
cls()
plt.close('all') 

#------------------------------------------------------------------------------
# Warnings
#------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# -----------------------------------------------------------------------------
# Transformation
# -----------------------------------------------------------------------------

def SOM2proflog10(X, SOM_ind2xy, SOM_codebook):
    X = np.round(X,0).astype('int')
    X[X<0]=0
    X[X[:,0]>np.max(SOM_lab2xy[:,0])] = np.max(SOM_lab2xy[:,0])
    X[X[:,1]>np.max(SOM_lab2xy[:,1])] = np.max(SOM_lab2xy[:,1])    
    tmp = np.zeros((np.shape(X)[0],np.shape(SOM_codebook)[1]))
    for i in range(0,np.shape(X)[0]):
        idx = np.where((SOM_ind2xy[:,0]==X[i,0]) & 
                       (SOM_ind2xy[:,1]==X[i,1]))[0]
        tmp[i] = np.log10(SOM_codebook[idx])
    return tmp   

#----------------------------------------------------------------------
# Read files
#----------------------------------------------------------------------

# Read File
nemo = np.load('/usr/home/mcane/Documents/Data/VectSorted.npy')
depth = sio.loadmat('/usr/home/mcane/Documents/Data/VectLB19922008.mat')['depth'][0:16]

# SOM
SOM_codebook = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout.mat')['SOM_codebook']
SOM_labels   = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout.mat')['SOM_labels'][0]
SOM_xy_coord = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout.mat')['SOM_xy_coord']
SOM_lab2xy   = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout.mat')['SOM_lab2xy']

nemo = lib.norm(nemo,-1,1)

#----------------------------------------------------------------------
# Training 
#----------------------------------------------------------------------

yearini = 1992
yearfin = 2005

scores_mse = []
scores_rms = []

for i in range(yearini,yearfin+1):
    
    print("\nval = {}\n".format(i))
    
    # Training Data Set
    x_train, y_train, Y_train, x_val, y_val, Y_val, x_test, y_test, Y_test = lib.kfold_SOM(nemo,SOM_xy_coord,yearini,yearfin,i,'all') 

    # Model
    model = Sequential()
    
    model.add(Dense(128, input_dim=5, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
    model.add(Dense(64, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))   
    
    model.add(Dense(2, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('linear'))
        
    #model.summary()
    
    # Training
    batch_size = 32
    nb_epoch = 5000
    
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')    
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    training = model.fit(x_train, Y_train,
                         validation_data=(x_val,Y_val),
                         batch_size=batch_size, epochs=nb_epoch,
                         verbose=1, callbacks=[earlyStopping])    
    
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='mse')    
    lib.training_plot(training.history)
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
    training = model.fit(x_train, Y_train,
                         validation_data=(x_val,Y_val),
                         batch_size=batch_size, epochs=nb_epoch,
                         verbose=1, callbacks=[earlyStopping])
    lib.training_plot(training.history)    
    
    # Prediction
    Y_pred = model.predict(x_test)    
    y_pred = SOM2proflog10(Y_pred, SOM_lab2xy, SOM_codebook) 
    
    mse = mean_squared_error(np.power(10,y_test), np.power(10,y_pred))
    scores_mse.append(mse)
    scores_rms.append(np.sqrt(mse))
    
    print("\nloss: sqrt(%.5f) = %.4f" % (mse, np.sqrt(mse))) 
    
    break
    
print("/n/n")      
for sc in range(0,np.shape(scores_mse)[0]): print(scores_mse[sc])
for sc in range(0,np.shape(scores_rms)[0]): print(scores_rms[sc])    