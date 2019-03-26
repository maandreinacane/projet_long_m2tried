# -*- coding: utf-8 -*-
"""
Descriptive Analysis
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

plt.close("all")

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------

# Read File
nemo = sio.loadmat('/usr/home/mcane/Documents/Data/VectLB19922008.mat')['Vect']
chl = sio.loadmat('/usr/home/mcane/Documents/Data/Vect_time.mat')['SCHL_ts']
sst = sio.loadmat('/usr/home/mcane/Documents/Data/Vect_time.mat')['ST_ts']
time = sio.loadmat('/usr/home/mcane/Documents/Data/Vect_time.mat')['Vect_time'][:,1:]
modis = np.vstack((time.T,chl,sst)).T
del chl, sst, time

# Variable Names
varname = ['SSH', 'CC', 'WS', 'SR', 'SST',
           'THERM 1', 'THERM 2', 'THERM 3', 'THERM 4', 'THERM 5','THERM 6', 
           'THERM 7', 'THERM 8', 'THERM 9', 'THERM 10', 'THERM 11', 'THERM 12',
           'THERM 13', 'THERM 14', 'THERM 15', 'THERM 16', 'THERM 17',  
           'CHL 1','CHL 2', 'CHL 3', 'CHL 4', 'CHL 5', 'CHL 6', 'CHL 7', 
           'CHL 8', 'CHL 9', 'CHL 10','CHL 11', 'CHL 12', 'CHL 13', 'CHL 14', 
           'CHL 15', 'CHL 16', 'CHL 17', 'CHL 18', 
           '5days', 'year', 'latitude', 'longitude']

var_inout = ['SSH', 'CC', 'WS', 'SR', 'SST',
                   'CHL 1','CHL 2', 'CHL 3', 'CHL 4', 'CHL 5', 'CHL 6', 'CHL 7', 
                   'CHL 8', 'CHL 9', 'CHL 10','CHL 11', 'CHL 12', 'CHL 13', 'CHL 14', 
                   'CHL 15', 'CHL 16']

# Selecting the BATS Region
idx = np.where((nemo[:,-1] <= -63) & (nemo[:,-1] >= -65) & 
               (nemo[:,-2] >= 31) & (nemo[:,-2] <= 33))[0]

nemo = nemo[idx]

# NEMO + MODIS
tmp = np.zeros((np.shape(nemo)[0],2))
for i in range(0,np.shape(nemo)[0]-5):
    idx = np.where((modis[:,0]==nemo[i,41]) & (modis[:,1]==nemo[i,40]))[0]  
    if np.shape(modis[idx])[0] == 1:
        tmp[i] = modis[idx,2:]
modis = tmp
del i, idx, tmp

data = np.hstack((nemo,modis))

#%%
# Creating Pandas' Data Frame
nemo = pd.DataFrame(nemo, columns=varname)

nemo['5days'] = nemo['5days'].astype('category')
nemo['year']  = nemo['year'].astype('category')
nemo['latitude'] = nemo['latitude'].astype('category')
nemo['longitude'] = nemo['longitude'].astype('category')

# Number of Categorical and Numerical Variables
ncv = 4
nnv = 40

# -----------------------------------------------------------------------------
# Simple Satistics
# -----------------------------------------------------------------------------
nemo_stats = nemo.describe()

# -----------------------------------------------------------------------------
# Linear Correlations
# -----------------------------------------------------------------------------
nemo_corr = nemo.corr()

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(nemo_corr)
plt.xticks(rotation=90)
ax1.set_xticks(np.arange(0, nnv, 1))
ax1.set_yticks(np.arange(0, nnv, 1))
ax1.set_xticklabels(varname[:-ncv],fontsize=6)
ax1.set_yticklabels(varname[:-ncv],fontsize=6)
fig.colorbar(cax)
plt.title('Correlation Matrix')
plt.show()

nemo_corr = nemo[['SSH', 'CC', 'WS', 'SR', 'SST',
                   'CHL 1','CHL 2', 'CHL 3', 'CHL 4', 'CHL 5', 'CHL 6', 'CHL 7', 
                   'CHL 8', 'CHL 9', 'CHL 10','CHL 11', 'CHL 12', 'CHL 13', 'CHL 14', 
                   'CHL 15', 'CHL 16']].corr()

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(nemo_corr)
plt.xticks(rotation=90)
ax1.set_xticks(np.arange(0, 21, 1))
ax1.set_yticks(np.arange(0, 21, 1))
ax1.set_xticklabels(var_inout,fontsize=6)
ax1.set_yticklabels(var_inout,fontsize=6)
clb = fig.colorbar(cax)
clb.ax.tick_params(labelsize=6)
plt.title('Matrice de Corrélation')
plt.show()


from pandas.plotting import scatter_matrix
scatter_matrix(nemo[['SSH', 'CC', 'WS', 'SR', 'SST',
                   'CHL 1','CHL 2', 'CHL 3', 'CHL 4', 'CHL 5', 'CHL 6', 'CHL 7', 
                   'CHL 8', 'CHL 9', 'CHL 10','CHL 11', 'CHL 12', 'CHL 13', 'CHL 14', 
                   'CHL 15', 'CHL 16']])
plt.title('Matrice de nuages de points')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Histograms
# -----------------------------------------------------------------------------
if 1:
    for i in range(0,len(nemo)):
        plt.figure()
        weights = np.ones_like(nemo[varname[i]])*1./len(nemo)
        nemo[varname[i]].hist(weights = weights, grid=False, color='#86bf91', zorder=2, rwidth=0.9)
        plt.ylabel('Fréquence')
        plt.title(varname[i])
        plt.show()
    del i

plt.figure(figsize=(20,3))
if 1:
    for i in range(0,5):
        plt.subplot(1,5,i+1)
        weights = np.ones_like(nemo[varname[i]])*1./len(nemo)
        nemo[varname[i]].hist(weights=weights, grid=False, color='#86bf91', zorder=2, rwidth=0.9)
        plt.ylabel('Fréquence')
        plt.title(varname[i])
        plt.ylim([0,0.25])
        plt.tight_layout()
        plt.show()
    del i
    
plt.figure(figsize=(20,5))
if 1:
    for i in range(22,38):
        plt.subplot(2,8,i-22+1)
        weights = np.ones_like(nemo[varname[i]])*1./len(nemo)
        nemo[varname[i]].hist(weights=weights, grid=False, color='#86bf91', zorder=2, rwidth=0.9)
        plt.ylabel('Fréquence')
        plt.title(varname[i])
        plt.ylim([0,0.6])
        plt.tight_layout()
        plt.show()
    del i    


del ncv, nnv
del varname

