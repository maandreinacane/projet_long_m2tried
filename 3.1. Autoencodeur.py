import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import math

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, UpSampling1D, Flatten, Reshape
from keras.callbacks import EarlyStopping

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


###-----------------------------------------------------------------------------------------------------------
##les fonctions utilisées :

def aleaGauss(sigma):
    U1 = random.random()
    U2 = random.random()
    return sigma*math.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2) 

def build_encoder(nc1, nc2, nd1, nd2, fs1, fs2):
        #partie codeur   
    x1 = Input(batch_shape=(None,16))
    x2 = Reshape((16,1))(x1)
    x3 = Conv1D(nc1,fs1, activation='relu',strides=2, padding='same')(x2)
    x5 = Conv1D(nc2,fs2, activation='relu',strides=2, padding='same' )(x3)
    x7 = Flatten()(x5)
    x8 = Dense(nd1,activation = 'linear')(x7)
    x9 = Dense(nd2,activation = 'linear')(x8)
       
        #partie decoder
    y1 = Reshape((nd2,1))(x9)
    y2 = Conv1D(nc2,fs2, activation='relu',padding='same')(y1)
    y3 = UpSampling1D(2)(y2)
    y4 = Conv1D(nc1,fs1, activation='relu',padding='same')(y3)
    y5 = UpSampling1D(2)(y4)
    y6 = Flatten()(y5)
    y7 = Dense(16,activation = 'linear')(y6)
   
        #autoencoder
    autoencoder = Model(input=x1, output=y7)
   
    return autoencoder
###-----------------------------------------------------------------------------------------------------------
    
##lecture des données
nemo = sio.loadmat('VectLB19922008.mat')['Vect']
    #réarangement des données dans le temps  
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

## données de chlorophylle 
chl=nemo[:,22:38]
chl=np.log10(chl)

#Standarize Data
chl = chl / np.max(abs(chl))


###-----------------------------------------------------------------------------------------------------------
## construire une entrée bruitée

N=np.shape(chl)
chl_noised = np.zeros(N)
chl_noised_test=np.zeros(N)
t = np.zeros(len(chl))
sigma = 0.005            ##choix du sigma pour la génération du bruit

for j in range(16):    
    for k in range(len(chl)):
        t[k] = k
        chl_noised[k,j] = chl[k,j]+aleaGauss(sigma)
        chl_noised_test[k,j]=chl[k,j]+aleaGauss(0.01)
        
##comparer les données normales et bruitées pour une profondeur de chlorophylle
plt.figure()
plt.plot(t,chl_noised[:,4])
plt.figure()
plt.plot(t,chl[:,4])

###-----------------------------------------------------------------------------------------------------------

## Training and Test Data Sets
indice_train = np.where((nemo[:,-3] <= 2005) & (nemo[:,-3] >= 1992))[0]
indice_test=np.where((nemo[:,-3] > 2005) & (nemo[:,-3] <= 2008) & (nemo[:,-1] <= -63) & (nemo[:,-1] >= -65) & 
               (nemo[:,-2] >= 31) & (nemo[:,-2] <= 33))[0]

x_train=chl_noised[indice_train,:]
y_train=chl[indice_train,:]

x_test=chl_noised_test[indice_test,:]
y_test=chl[indice_test,:]


##construction de l'autoencodeur

autoencoder=build_encoder(nc1=14, nc2=10, nd1=18, nd2=6, fs1=6, fs2=3) ##construction
autoencoder.compile(optimizer='adam', loss='mse')   #compiler

##  Apprentissage

batch_size = 128
nb_epoch = 500
earlyStopping=EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
training=autoencoder.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.1, callbacks=[earlyStopping])

##crbe d'erreur de validation et training
plt.figure()
for key in training.history:   
    plt.plot(training.history[key], label=key)
    plt.legend()
plt.show()


##prediction :
y_pred=autoencoder.predict(x_test)

##tracé de figures :

prof=np.arange(0,len(x_test[0,:]),1)
depth = sio.loadmat('VectLB19922008.mat')['depth'][0:16]
depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
depth = np.round(depth,0).astype('int') * (-1)
depth = np.flip(depth)

fig=plt.figure()

Se=[-73,72,-58,218,12,1,0,3,2,8]
for i in range(len(Se)) :
    
    ax1=plt.subplot(2,5,i+1)
    plt.plot(y_test[i,:],prof,'b')
    plt.plot(y_pred[i,:],prof,'r')
#    plt.plot(x_test[i,:],prof,'g',label="entrée bruitée")
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticks(np.arange(-0.8,-0.2, 0.1))
    
    del ax1
    
plt.ylabel('depth(m)', fontsize=6)
plt.xlabel('concentration de la chlorophyle')


##sauvegarde des poids pour les fixer à un autre training
weight=[]
for layer in autoencoder.layers:
    poids=layer.get_weights()
    weight.append(poids)
    
#np.save('poids régulariser autoencodeur.npy', weight)
