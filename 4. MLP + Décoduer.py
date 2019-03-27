#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:34:29 2019

@author: lchibane

réseau avec couches dense et décodeur 
entrée : les données de surface
sortie: les profils verticaux de la chlorophylle
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


from keras.models import Model, model_from_yaml, Sequential
from keras.layers import Input, Dense, Conv1D, UpSampling1D, Flatten, Reshape
from keras.callbacks import EarlyStopping, Callback
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from keras import layers
import library as lib1


nemo = sio.loadmat('VectLB19922008.mat')['Vect']

days = np.unique(nemo[:,40])
years = np.unique(nemo[:,41])

tmp = np.zeros(np.shape(nemo))
cont = 0
for year in years:
    for day in days:
        idx = np.where((nemo[:,41]==year) & (nemo[:,40]==day))[0]
        for region in range(0,np.shape(idx)[0]):
            tmp[cont] = nemo[idx[0]+region]
            cont = cont+1
nemo = tmp
del years, days, day, year, idx, cont, tmp

##data
chl=nemo[:,22:38]
surface_data=nemo[:,0:5]
depth = sio.loadmat('VectLB19922008.mat')['depth'][0:16]



## Training and Test Data Sets
indice_train = np.where((nemo[:,-3] <= 2005) & (nemo[:,-3] >= 1992))[0]
indice_test=np.where((nemo[:,-3] > 2005) & (nemo[:,-3] <= 2008) & (nemo[:,-1] <= -63) & (nemo[:,-1] >= -65) & 
               (nemo[:,-2] >= 31) & (nemo[:,-2] <= 33))[0]

x_train=surface_data[indice_train,:]
y_train=chl[indice_train,:]

x_test=surface_data[indice_test,:]
y_test=chl[indice_test,:]

## importation des poids
#poids=np.load('weight.npy')   #autoencodeur avec entrée normal
poids=np.load('poids régulariser autoencodeur.npy')   ##poids de l'autoencodeur avec entrée bruitée

## construiction du modèle :
##parametre
nd1=128
nd2=6
nc2=10
fs2=3
nc1=14
fs1=6

#FIRST TRAINING :
#réseau dense :
x1 = Input(batch_shape=(None,5))
x2 = Dense(nd1,activation = 'sigmoid')(x1)
x3 = Dense(nd2,activation = 'linear',trainable=True)(x2)
#x4 = Dense(nd2,activation = 'linear',trainable=True)(x3)
#décodeur
y1 = Reshape((nd2,1))(x3)    
y2 = Conv1D(nc2,fs2, activation='relu',padding='same',trainable=False)(y1)
y3 = UpSampling1D(2)(y2)    
y4 = Conv1D(nc1,fs1, activation='relu',padding='same',trainable=False)(y3)    
y5 = UpSampling1D(2)(y4)    
y6 = Flatten()(y5)    
y7 = Dense(16,activation = 'linear',trainable=False)(y6)

reseau1 = Model(input=x1, output=y7)
    
reseau1.layers[4].set_weights(poids[8])
reseau1.layers[6].set_weights(poids[10])
reseau1.layers[9].set_weights(poids[13])


reseau1.compile(optimizer='adam', loss='mse')
reseau1.summary()
nb_epoch=1000
batch_size=128
#earlyStopping=EarlyStoppingByLossVal(monitor='val_loss', value=0.008, verbose=1)
earlyStopping=EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='auto')
reseau1.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2, callbacks=[earlyStopping])

#SAVE WEIGHTS
w=[]
for layer in reseau1.layers:
    p=layer.get_weights()
    w.append(p)
    
##SECOND TRAINING
    #réseau dense
x1 = Input(batch_shape=(None,5))
x2 = Dense(nd1,activation = 'sigmoid',trainable=True)(x1)
x3 = Dense(nd2,activation = 'linear',trainable=True)(x2)
#x4 = Dense(nd2,activation = 'linear',trainable=True)(x3)

    #décodeur
y1 = Reshape((nd2,1))(x3)    
y2 = Conv1D(nc2,fs2, activation='relu',padding='same',trainable=True)(y1)
y3 = UpSampling1D(2)(y2)    
y4 = Conv1D(nc1,fs1, activation='relu',padding='same',trainable=True)(y3)    
y5 = UpSampling1D(2)(y4)    
y6 = Flatten()(y5)    
y7 = Dense(16,activation = 'linear',trainable=True)(y6)

reseau1 = Model(input=x1, output=y7)

reseau1.layers[1].set_weights(w[1])
reseau1.layers[2].set_weights(w[2]) 
reseau1.layers[4].set_weights(w[4])
reseau1.layers[6].set_weights(w[6])
reseau1.layers[9].set_weights(w[9])


reseau1.compile(optimizer='adam', loss='mse')
reseau1.summary()
nb_epoch=1000
batch_size=128
#earlyStopping=EarlyStoppingByLossVal(monitor='val_loss', value=0.008, verbose=1)
earlyStopping=EarlyStopping(monitor='val_loss', patience=100, verbose=2, mode='auto')
reseau1.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.2, callbacks=[earlyStopping])

    #tester les performances
y_pred=reseau1.predict(x_test)
# Obtaining the Last Year 2008        
y_test_2008 = y_test[-73:]  
y_pred_2008 = y_pred[-73:]  

# Obtaining the color levels
lev_vec = lib1.get_levs_log(y_test_2008, levs=20)

# Replace extreme low values
y_pred_2008[y_pred_2008 <= np.exp(np.min(lev_vec))] = np.exp(np.min(lev_vec))

##visualisation
lib1.display_year_log(y_test_2008, depth, lev_vec, title='y_test (2008)')      
lib1.display_year_log(y_pred_2008, depth, lev_vec, title='y_pred (2008)')  

ww=[]
for layer in reseau1.layers:
    p=layer.get_weights()
    ww.append(p)
    
#mse = mean_squared_error(np.power(10,y_test), np.power(10,y_pred))  #calcul de l'erreur qui ns importe
mse = mean_squared_error(y_test_2008,y_pred_2008)
print(np.sqrt(mse))
    
#np.save('poids_reseau1.npy', ww)