#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lib
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def norm(data,vmin,vmax):
    scaler = MinMaxScaler(feature_range=(vmin, vmax))
    data[:,:5] = scaler.fit_transform(data[:,:5])  
    return data

def kfold(data, yearini, yearfin, yearval, flag):
    if flag=='all':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval))[0]
    if flag=='BATS':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval) &
                             (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                             (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]   
    idx_test = np.where((data[:,-3] == yearval) &
                        (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                        (data[:,-2] >= 31) & (data[:,-2] <= 33))[0] 
    x_train = data[idx_train,:5]
    y_train = np.log10(data[idx_train,22:38])
    x_test= data[idx_test,:5]
    y_test = np.log10(data[idx_test,22:38])    
    if flag=='all':
        x_val = x_train[-73*9:]
        y_val = y_train[-73*9:]    
        x_train = x_train[:-73*9]
        y_train = y_train[:-73*9]
    if flag=='BATS':
        x_val = x_train[-73:]
        y_val = y_train[-73:]    
        x_train = x_train[:-73]
        y_train = y_train[:-73]        
    return x_train, y_train, x_val, y_val, x_test, y_test

def kfold_SOM(data, SOM_xy_coord, yearini, yearfin, yearval, flag):
    if flag=='all':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval))[0]
    if flag=='BATS':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval) &
                             (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                             (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]   
    idx_test = np.where((data[:,-3] == yearval) &
                        (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                        (data[:,-2] >= 31) & (data[:,-2] <= 33))[0] 
    x_train = data[idx_train,:5]
    y_train = np.log10(data[idx_train,22:38])
    Y_train = SOM_xy_coord[idx_train]
    x_test= data[idx_test,:5]
    y_test = np.log10(data[idx_test,22:38]) 
    Y_test = SOM_xy_coord[idx_test]
    if flag=='all':
        x_val = x_train[-73*9:]
        y_val = y_train[-73*9:]  
        Y_val = Y_train[-73*9:]
        x_train = x_train[:-73*9]
        y_train = y_train[:-73*9]
        Y_train = Y_train[:-73*9]
    if flag=='BATS':
        x_val = x_train[-73:]
        y_val = y_train[-73:] 
        Y_val = Y_train[-73:]
        x_train = x_train[:-73]
        y_train = y_train[:-73]  
        Y_train = Y_train[:-73]
    return x_train, y_train, Y_train, x_val, y_val, Y_val, x_test, y_test, Y_test

def kfold_PCA(data, y_pca, yearini, yearfin, yearval, flag):
    if flag=='all':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval))[0]
    if flag=='BATS':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval) &
                             (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                             (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]   
    idx_test = np.where((data[:,-3] == yearval) &
                        (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                        (data[:,-2] >= 31) & (data[:,-2] <= 33))[0] 
    x_train = data[idx_train,:5]
    y_train = np.log10(data[idx_train,22:38])
    Y_train = y_pca[idx_train]
    x_test= data[idx_test,:5]
    y_test = np.log10(data[idx_test,22:38]) 
    Y_test = y_pca[idx_test]
    if flag=='all':
        x_val = x_train[-73*9:]
        y_val = y_train[-73*9:]  
        Y_val = Y_train[-73*9:]
        x_train = x_train[:-73*9]
        y_train = y_train[:-73*9]
        Y_train = Y_train[:-73*9]
    if flag=='BATS':
        x_val = x_train[-73:]
        y_val = y_train[-73:] 
        Y_val = Y_train[-73:]
        x_train = x_train[:-73]
        y_train = y_train[:-73]  
        Y_train = Y_train[:-73]
    return x_train, y_train, Y_train, x_val, y_val, Y_val, x_test, y_test, Y_test

def split_train_val(data, yearini, yearfin, yearval, flag):
    if flag=='all':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval))[0]
        idx_val = np.where(data[:,-3] == yearval)[0]
    if flag=='BATS':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval) &
                             (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                             (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]   
        idx_val = np.where((data[:,-3] == yearval) &
                            (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                            (data[:,-2] >= 31) & (data[:,-2] <= 33))[0] 
    x_train = data[idx_train,:5]
    y_train = np.log10(data[idx_train,22:38])
    x_val= data[idx_val,:5]
    y_val = np.log10(data[idx_val,22:38])         
    return x_train, y_train, x_val, y_val

def split_train_val_SOM(data, SOM_xy_coord, yearini, yearfin, yearval, flag):
    if flag=='all':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval))[0]
        idx_val = np.where(data[:,-3] == yearval)[0]
    if flag=='BATS':
        idx_train = np.where((data[:,-3] >= yearini) &
                             (data[:,-3] <= yearfin) &
                             (data[:,-3] != yearval) &
                             (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                             (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]   
        idx_val = np.where((data[:,-3] == yearval) &
                            (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                            (data[:,-2] >= 31) & (data[:,-2] <= 33))[0] 
    x_train = data[idx_train,:5]
    y_train = np.log10(data[idx_train,22:38])
    Y_train = SOM_xy_coord[idx_train]
    x_val = data[idx_val,:5]
    y_val = np.log10(data[idx_val,22:38]) 
    Y_val = SOM_xy_coord[idx_val]
    return x_train, y_train, Y_train, x_val, y_val, Y_val

def split_test(data, yearini, yearfin, flag):
    if flag=='all':
        idx = np.where((data[:,-3] >= yearini) &
                       (data[:,-3] <= yearfin))[0]
    if flag=='BATS':
        idx = np.where((data[:,-3] >= yearini) &
                       (data[:,-3] <= yearfin) &
                       (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                       (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]           
    x_test = data[idx,:5]
    y_test = np.log10(data[idx,22:38])
    return x_test, y_test

def split_test_SOM(data, SOM_xy_coord, yearini, yearfin, flag):
    if flag=='all':
        idx = np.where((data[:,-3] >= yearini) &
                       (data[:,-3] <= yearfin))[0]
    if flag=='BATS':
        idx = np.where((data[:,-3] >= yearini) &
                       (data[:,-3] <= yearfin) &
                       (data[:,-1] <= -63) & (data[:,-1] >= -65) & 
                       (data[:,-2] >= 31) & (data[:,-2] <= 33))[0]           
    x_test = data[idx,:5]
    y_test = np.log10(data[idx,22:38])    
    Y_test = SOM_xy_coord[idx]
    return x_test, y_test, Y_test

def training_plot(training_history):
    plt.figure()
    for key in training_history:    
        plt.plot(training_history[key], label=key)
        plt.legend()
    plt.show()

def get_levs(X, levs=20):
    # Levels for the Contour Visualization
    lev_min = np.floor(X.min()*10)/10
    lev_max = np.ceil(X.max()*10)/10
    lev_step = (lev_max-lev_min)/levs
    lev_vec = np.arange(lev_min, lev_max+lev_step, lev_step)
    return lev_vec

def get_levs_log(X, levs=20):
    # Levels for the Contour Visualization
    lev_min = np.floor(np.log10(X.min()))
    lev_max = np.ceil(np.log10(X.max()))
    lev_vec = np.arange(lev_min, lev_max, (lev_max-lev_min)/levs)
    return lev_vec

def display_seq(X, years, depth, levs=20):

    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)
    
    # Levels for the Contour Visualization
    lev_min = np.floor(np.log10(X.min()))
    lev_max = np.ceil(np.log10(X.max()))
    lev_vec = np.arange(lev_min, lev_max, (lev_max-lev_min)/levs)
        
    X = np.flip(X, axis=1) 
    depth = np.flip(depth, axis=0)   
    
    # Logarithm
    Xlog = np.log10(X)
    
    # Figure All Years
    fig = plt.figure(figsize=(17,4))
    ax1 = fig.add_subplot(111)
    cont = plt.contourf(Xlog.T,cmap='jet', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, np.shape(np.unique(years))[0]*73, 73))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(np.unique(years).astype('int'),fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Concentration de Chlorophylle-A (ng/l)', rotation=270, labelpad=10, fontsize=6)    
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()

def display_year(X, depth, lev_vec, title=None):
    
    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)
        
    X = np.flip(X, axis=1) 
    depth = np.flip(depth, axis=0)    
    
    # Figure one year
    fig = plt.figure(figsize=(17,4))
    ax1 = fig.add_subplot(111)
    cont = ax1.contourf(X.T,cmap='jet', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, 12*6.1, 6.1))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(['Janv','Fév','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc'],fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    if title != None:
        plt.title(title,fontsize=8)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Concentration de Chlorophylle-A (ng/l)', rotation=270, labelpad=10, fontsize=6)
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()    
    
def display_year_log(X, depth, lev_vec, title=None):
    
    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)
        
    X = np.flip(X, axis=1) 
    depth = np.flip(depth, axis=0)
    
    # Logarithm
    Xlog = np.log10(X)
    
    # Figure one year
    fig = plt.figure(figsize=(17,4))
    ax1 = fig.add_subplot(111)
    cont = ax1.contourf(Xlog.T,cmap='jet', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, 12*6.1, 6.1))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(['Janv','Fév','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc'],fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    if title != None:
        plt.title(title,fontsize=8)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Concentration de Chlorophylle-A (ng/l)', rotation=270, labelpad=10, fontsize=6)
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()

def display_dif(X, depth, lev_vec, title=None):
    
    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)
        
    X = np.flip(X, axis=1) 
    depth = np.flip(depth, axis=0)    
    
    # Figure one year
    fig = plt.figure(figsize=(17,4))
    ax1 = fig.add_subplot(111)
    cont = ax1.contourf(X.T,cmap='Blues', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, 12*6.1, 6.1))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(['Janv','Fév','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc'],fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    if title != None:
        plt.title(title,fontsize=8)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Erreur absolue (ng/l)', rotation=270, labelpad=10, fontsize=6)
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()       
    
def display_3years(X, depth, lev_vec, title=None):
    
    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)
        
    X = np.flip(X, axis=1) 
    depth = np.flip(depth, axis=0)    
    
    # Figure one year
    fig = plt.figure(figsize=(20,2))
    ax1 = fig.add_subplot(111)
    cont = ax1.contourf(X.T,cmap='jet', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, 219, (73/12)))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(['Janv','Fév','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc']*3,fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    if title != None:
        plt.title(title,fontsize=8)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Concentration de Chlorophylle-A (ng/l)', rotation=270, labelpad=10, fontsize=6)
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()   
    
def display_3yearsdif(X, depth, lev_vec, title=None):
    
    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)
        
    X = np.flip(X, axis=1) 
    depth = np.flip(depth, axis=0)    
    
    # Figure one year
    fig = plt.figure(figsize=(20,2))
    ax1 = fig.add_subplot(111)
    cont = ax1.contourf(X.T,cmap='Blues', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, 12*6.1*3, 6.1))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(['Janv','Fév','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc']*3,fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    if title != None:
        plt.title(title,fontsize=8)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Erreur absolue (ng/l)', rotation=270, labelpad=10, fontsize=6)
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()     
    
def display_avyear(X, days, years, depth, levs=20):
    Xm = np.zeros((np.shape(np.unique(days))[0],np.shape(X)[1]))
    # Averaging days
    for i in range(0,np.shape(np.unique(days))[0]):
        idx = np.where(days==i+1)    
        Xm[i] = np.mean(X[idx], axis=0)
    
    # Depth
    depth = np.array(depth).reshape(1,np.shape(depth)[0])[0]
    depth = np.round(depth,0).astype('int') * (-1)

    # Levels for the Contour Visualization
    lev_min = np.floor(np.log10(X.min()))
    lev_max = np.ceil(np.log10(X.max()))
    lev_vec = np.arange(lev_min, lev_max, (lev_max-lev_min)/levs)
      
    Xm = np.flip(Xm, axis=1) 
    depth = np.flip(depth, axis=0)   
    
    # Logarithm
    Xmlog = np.log10(Xm)
    
    # Figure All Years
    fig = plt.figure(figsize=(17,4))
    ax1 = fig.add_subplot(111)
    cont = plt.contourf(Xmlog.T,cmap='jet', corner_mask=True, levels=lev_vec)
    ax1.set_yticks(np.arange(0, np.shape(depth)[0], 1))
    ax1.set_xticks(np.arange(0, 12*6.1, 6.1))
    ax1.set_yticklabels(depth,fontsize=6)
    ax1.set_xticklabels(['Janv','Fév','Mars','Avr','Mai','Juin','Juil','Août','Sept','Oct','Nov','Déc'],fontsize=6)
    plt.ylabel('Profondeur (m)', fontsize=6)
    clb = fig.colorbar(cont, ax=ax1)
    clb.set_label('Concentration de Chlorophylle-A (ng/l)', rotation=270, labelpad=10, fontsize=6)    
    clb.ax.tick_params(labelsize=6)
    plt.grid(which='minor')
    plt.show()    