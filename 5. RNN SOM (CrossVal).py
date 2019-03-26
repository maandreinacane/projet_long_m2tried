#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN Many-to-Many SOM (CrossVal)
"""

import library as lib

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import InputLayer, LSTM, GRU, Bidirectional, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error

#------------------------------------------------------------------------------
# Transformations
#------------------------------------------------------------------------------

def sliding_window_tranf(data, n_steps, flag='all'):
    if flag=='all':
        tmp = np.zeros((np.shape(data)[0]-n_steps+1,n_steps,np.shape(data)[1]))        
        for i in range(0,np.shape(data)[0]-n_steps+1):
            for j in range(0,n_steps):
                for k in range(0,np.shape(data)[1]):
                    tmp[i,j,k] = data[i+j,k]   
    if flag=='final':
        tmp = np.zeros((np.shape(data)[0]-n_steps+1,np.shape(data)[1]))  
        for i in range(0,np.shape(data)[0]-n_steps+1):
            j= n_steps-1
            for k in range(0,np.shape(data)[1]):
                tmp[i,k] = data[i+j,k]     
    if flag=='div':
        divs = int(np.floor(np.shape(data)[0]/n_steps))
        tmp = np.zeros((divs,n_steps,np.shape(data)[1]))        
        for i in range(0,divs*n_steps,n_steps):
            tmp[int(i/n_steps)] = data[i:i+n_steps]            
    return tmp

def sliding_window_inv(data):
    tmp = np.zeros((np.shape(data)[0],np.shape(data)[0]+np.shape(data)[1]-1,np.shape(data)[2]))
    for i in range(0,np.shape(data)[0]):
        for j in range(0,np.shape(data)[1]):
            for k in range(0,np.shape(data)[2]):
                tmp[i,j+i,k] = data[i,j,k]  
    tmp[tmp == 0] = np.nan            
    tmp_flat = np.zeros((np.shape(data)[0]+np.shape(data)[1]-1,np.shape(data)[2]))
    for i in range(0,np.shape(tmp)[1]):
        for j in range(0,np.shape(tmp)[2]):
            tmp_flat[i,j] = np.nanmedian(tmp[:,i,j])
    tmp_flat[np.isnan(tmp_flat)] = 0            
    return tmp_flat

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

#------------------------------------------------------------------------------
# Read files
#------------------------------------------------------------------------------

# Read File
nemo = np.load('/usr/home/mcane/Documents/Data/VectSorted.npy')
depth = sio.loadmat('/usr/home/mcane/Documents/Data/VectLB19922008.mat')['depth'][0:16]

# SOM
SOM_codebook = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout_v3.mat')['SOM_codebook']
SOM_labels   = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout_v3.mat')['SOM_labels'][0]
SOM_xy_coord = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout_v3.mat')['SOM_xy_coord']
SOM_lab2xy   = sio.loadmat('/usr/home/mcane/Documents/Data/SOMout_v3.mat')['SOM_lab2xy']

# Input Normalization
nemo = lib.norm(nemo,-1,1)

#----------------------------------------------------------------------
# Training 
#----------------------------------------------------------------------

yearini = 1992
yearfin = 2005

scores_mse = []
scores_rms = []

n_steps = 10
n_var_in = 5
n_var_out = 2

for i in range(yearini,yearfin+1):
    
    print("\nval = {}\n".format(i))
    
    # Training, Val and Test Data Sets    
    x_train, y_train, y_train_som, x_val, y_val, y_val_som, x_test, y_test, y_test_som = lib.kfold_SOM(nemo,SOM_xy_coord,yearini,yearfin,i,'all') 
   
    X_train = sliding_window_tranf(x_train, n_steps, 'all')          
    Y_train = sliding_window_tranf(y_train_som, n_steps, 'all')   
    
    X_val = sliding_window_tranf(x_val, n_steps, 'all')          
    Y_val = sliding_window_tranf(y_val_som, n_steps, 'all') 
    
    X_test = sliding_window_tranf(x_test, n_steps, 'all')          
    Y_test = sliding_window_tranf(y_test_som, n_steps, 'all')    
            
    # Model
    model = Sequential()
    model.add(InputLayer(input_shape=(n_steps,n_var_in)))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(n_var_out, activation='linear')))
    model.summary()
    
    # Training    
    batch_size = 32
    nb_epoch = 500 
    
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')     
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    training = model.fit(X_train, Y_train,
                         validation_data=(X_val,Y_val),
                         batch_size=batch_size, epochs=nb_epoch,
                         verbose=1, callbacks=[earlyStopping])
    lib.training_plot(training.history)        

    # Prediction
    Y_pred = model.predict(X_test)
    y_pred_som = sliding_window_inv(Y_pred)
    y_pred = SOM2proflog10(y_pred_som, SOM_lab2xy, SOM_codebook) 
    
    mse = mean_squared_error(np.power(10,y_test), np.power(10,y_pred))
    scores_mse.append(mse)
    scores_rms.append(np.sqrt(mse))
    
    print("\nloss: sqrt(%.5f) = %.4f" % (mse, np.sqrt(mse)))           
