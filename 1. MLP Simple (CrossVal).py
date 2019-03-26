#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP Simple (CrossVal)
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
# Read files
#----------------------------------------------------------------------

# Read File
nemo = np.load('/usr/home/mcane/Documents/Data/VectSorted.npy')
depth = sio.loadmat('/usr/home/mcane/Documents/Data/VectLB19922008.mat')['depth'][0:16]

# Input Normalization
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
    
    # Training, Val and Test Data Sets
    x_train, y_train, x_val, y_val, x_test, y_test = lib.kfold(nemo,yearini,yearfin,i,'all')    
    
    # Model
    model = Sequential()
    
    model.add(Dense(64, input_dim=5, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
    model.add(Dense(128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))    
    
    model.add(Dense(16, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('linear'))
        
    #model.summary()
    
    # Training
    batch_size = 32
    nb_epoch = 5000
    
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse')    
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    training = model.fit(x_train, y_train,
                         validation_data=(x_val,y_val),
                         batch_size=batch_size, epochs=nb_epoch,
                         verbose=1, callbacks=[earlyStopping])    
    
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss='mse')    
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    training = model.fit(x_train, y_train,
                         validation_data=(x_val,y_val),
                         batch_size=batch_size, epochs=nb_epoch,
                         verbose=0, callbacks=[earlyStopping])
    lib.training_plot(training.history)    
    
    # Prediction
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(np.power(10,y_test), np.power(10,y_pred))
    scores_mse.append(mse)
    scores_rms.append(np.sqrt(mse))
    
    print("\nloss: sqrt(%.5f) = %.4f" % (mse, np.sqrt(mse)))  
