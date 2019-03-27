import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import InputLayer, LSTM, GRU, Bidirectional, TimeDistributed, Dense, Reshape, Flatten, UpSampling1D, Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback

from sklearn.metrics import mean_squared_error

import La_lib as lib

#------------------------------------------------------------------------------
# Read files
#------------------------------------------------------------------------------

#np.random.seed(0)
# Read File
nemo = np.load('VectSorted.npy')
depth = sio.loadmat('VectLB19922008.mat')['depth'][0:16]

nemo = lib.norm(nemo,-1,1)


# Training Data Set
yearini = 1992
yearfin = 2005
yearval = 1999

x_train, y_train, x_val, y_val = lib.split_train_val(nemo,yearini,yearfin,yearval,'all')  
x_test, y_test = lib.split_test(nemo,2006,2008,'BATS')

#------------------------------------------------------------------------------
# Sliding window transformation
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
            tmp_flat[i,j] = np.nanmean(tmp[:,i,j])
    tmp_flat[np.isnan(tmp_flat)] = 0            
    return tmp, tmp_flat

n_steps = 10
n_var_in = np.shape(x_train)[1]
n_var_out = np.shape(y_train)[1] 

X_train = sliding_window_tranf(x_train, n_steps, 'all')          
Y_train = sliding_window_tranf(y_train, n_steps, 'all')   

X_val = sliding_window_tranf(x_val, n_steps, 'all')          
Y_val = sliding_window_tranf(y_val, n_steps, 'all') 

X_test = sliding_window_tranf(x_test, n_steps, 'all')          
Y_test = sliding_window_tranf(y_test, n_steps, 'all') 

#------------------------------------------------------------------------------
# Model
#------------------------------------------------------------------------------
poids=np.load('poids régulariser autoencodeur.npy')   ##poids de l'autoencodeur avec entrée bruitée



nd2=6
nc2=10
fs2=3
nc1=14
fs1=6

#First Train
model = Sequential()
model.add(InputLayer(input_shape=(n_steps,n_var_in)))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(TimeDistributed(Dense(64,activation = 'linear',trainable=True)))
model.add(TimeDistributed(Dense(6,activation = 'linear',trainable=True)))


#décodeur
model.add(TimeDistributed(Reshape((nd2,1)))   ) 
model.add(TimeDistributed(Conv1D(nc2,fs2, activation='relu',padding='same'),trainable=False))
model.add(TimeDistributed(UpSampling1D(2)))    
model.add(TimeDistributed(Conv1D(nc1,fs1, activation='relu',padding='same'),trainable=False))    
model.add(TimeDistributed(UpSampling1D(2))   ) 
model.add(TimeDistributed(Flatten())    )
model.add(TimeDistributed(Dense(n_var_out, activation='linear'),trainable=False))
model.summary()


model.layers[4].set_weights(poids[8])
model.layers[6].set_weights(poids[10])
model.layers[9].set_weights(poids[13])



#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

batch_size = 32
nb_epoch = 500

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse')   

earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
training = model.fit(X_train, Y_train,
                     validation_data=(X_val,Y_val),
                     batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, callbacks=[earlyStopping])


##save weights

w=[]
for layer in model.layers:
    p=layer.get_weights()
    w.append(p)
    
    
###Second Train
model = Sequential()
model.add(InputLayer(input_shape=(n_steps,n_var_in)))
model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(TimeDistributed(Dense(64,activation = 'linear',trainable=True)))
model.add(TimeDistributed(Dense(6,activation = 'linear',trainable=True)))


#décodeur
model.add(TimeDistributed(Reshape((nd2,1)))   ) 
model.add(TimeDistributed(Conv1D(nc2,fs2, activation='relu',padding='same')))
model.add(TimeDistributed(UpSampling1D(2)))    
model.add(TimeDistributed(Conv1D(nc1,fs1, activation='relu',padding='same'))   ) 
model.add(TimeDistributed(UpSampling1D(2))   ) 
model.add(TimeDistributed(Flatten())    )
model.add(TimeDistributed(Dense(n_var_out, activation='linear')))
model.summary()

 

#
model.layers[0].set_weights(w[0])

model.layers[1].set_weights(w[1])
model.layers[2].set_weights(w[2])
model.layers[4].set_weights(w[4])
model.layers[6].set_weights(w[6])
model.layers[9].set_weights(w[9])

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse') 
#
earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
training = model.fit(X_train, Y_train,
                     validation_data=(X_val,Y_val),
                     batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, callbacks=[earlyStopping])
### Weights
w=[]
for layer in model.layers:
    p=layer.get_weights()
    w.append(p)

# Prediction
Y_pred = model.predict(X_test)
tmp, y_pred = sliding_window_inv(Y_pred)
#
# Generalization Error
mse = mean_squared_error(np.power(10,y_test), np.power(10,y_pred))  
print("\nloss: sqrt(%.5f) = %.4f" % (mse, np.sqrt(mse))) 

## 10% Max and Min Error
idx_min = (y_test < np.percentile(y_test, 10))
idx_max = (y_test > np.percentile(y_test, 90))

mse_min = mean_squared_error(np.power(10,y_test[idx_min]), np.power(10,y_pred[idx_min]))
mse_max = mean_squared_error(np.power(10,y_test[idx_max]), np.power(10,y_pred[idx_max]))

print("\nloss min: sqrt(%.5f) = %.4f" % (mse_min, np.sqrt(mse_min)))  
print("\nloss max: sqrt(%.5f) = %.4f" % (mse_max, np.sqrt(mse_max)))  
##
# Generalization Error 2008
y_test_2008 = y_test[-73:]
y_pred_2008 = y_pred[-73:]

mse2008 = mean_squared_error(np.power(10,y_test_2008), np.power(10,y_pred_2008))
print("\nloss 2008: sqrt(%.5f) = %.4f" % (mse2008, np.sqrt(mse2008))) 

# 10% Max and Min Error
idx_min_2008 = (y_test_2008 < np.percentile(y_test_2008, 10))
idx_max_2008 = (y_test_2008 > np.percentile(y_test_2008, 90))

mse_min_2008 = mean_squared_error(np.power(10,y_test_2008[idx_min_2008]), np.power(10,y_pred_2008[idx_min_2008]))
mse_max_2008 = mean_squared_error(np.power(10,y_test_2008[idx_max_2008]), np.power(10,y_pred_2008[idx_max_2008]))

print("\nloss min 2008: sqrt(%.5f) = %.4f" % (mse_min_2008, np.sqrt(mse_min_2008)))  
print("\nloss max 2008: sqrt(%.5f) = %.4f" % (mse_max_2008, np.sqrt(mse_max_2008)))  
##
###------------------------------------------------------------------------------
### Visualizations
###------------------------------------------------------------------------------
   
lev_vec = lib.get_levs(y_test, levs=16)
lib.display_3years(y_test, depth, lev_vec,
                   title='Valeurs réelles de Chlorophylle-A (2006 à 2008)')
lib.display_3years(y_pred, depth, lev_vec,
                   title='Valeurs reconstituées de Chlorophylle-A (2006 à 2008)')

y_dif = abs(np.power(10,y_test)-np.power(10,y_pred))
lev_vec = lib.get_levs(y_dif, levs=10)
lev_vec = np.arange(0, 0.6+0.06, 0.06)
lib.display_3yearsdif(y_dif, depth, lev_vec,
                      title='Erreur absolue entre les valeurs réelles et reconstituéees (2006 à 2008)')

##------------------------------------------------------------------------------
## Save the weights
##------------------------------------------------------------------------------
#
np.save('weights_RNN+déc.npy',w)