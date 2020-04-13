# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:59:51 2020

@author: Akshay kumar C P
"""

# Self organising map ( SOM )

'''
Business problem : fraud detection

Deep learning scientist given an dataset customers from bank. 

usually customer's fill application form while aplying for credit card.
out mission is to detech the potencial fraud from application
end of the mission we need to give explicit list of customer who potentially cheated
'''

# unsupervised deep learning


# import lib's

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

# imnport dataset
# dataset is taken from UCI ML repo - statlog ( Australian credit approval ) data set

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# can distinguish btwen aproval not not aproved so divinding into X nd y
# only on X train

# feature scalling 
# for computing fast, high computation to make , 0-1 values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# train SOM - train from srach or copy from other dev

from minisom import MiniSom
# dim shud not be too small. in data set we don't ve mcuch customer of data so 10 X 10. so 
# sigma is the radius 1.0 . input length = 14 + 1 = 15. learning rate would be low.

som = MiniSom(x = 10, y =10,input_len=15,sigma = 1.0, learning_rate= 0.5)

# train som on X, before that initiaze weights

som.random_weights_init(X)

som.train_random(X,num_iteration=100)

# visualize the results

# plot som, we gona se 2D grid that contain all final wining node. from each wining node we get MID mean into neuron distance. 
# The MID for specific wining node is the mean of distances of all the neuron around the wining node.. the higher is the MID then the wining node is the far away from its neighbour.  the higher is MID from the neighbour then it's outlier
# majority of wining node represts the rules are respected 
# An outlier neuron far from majority os neuron is therefore far from the general rules - is how we detect outlier i,e frauds
# each neuron will get the MID, so we'll need to take the winning nodes that have higher MID and look at color - closer to white

from pylab import bone, pcolor,colorbar,plot,show

# initiaze the figure i,e window that contain map
bone()
# next,first - put the different wining nodes on map. add map infor of mean into neuron distance for all wining nodes the som identified.
# use colors for differnect range values of mean into neuron distances
pcolor(som.distance_map().T)
# methods returns the matrix of all the distances from the wining node. to get right order for pcolor function transpose the matrix

# add legends to know
colorbar()

# add markers on map
# 2 markets = red - no approval.

markers = ['o','s']
colors = ['r','g']

# no association yet

for i,x in enumerate(X):
    # get wining node for every customer
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         # 0.5 to place at middle.. y[i] = 0 = 'o'
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2)
show()
    
# white - fraud

# explicitly acpture frauds
# finding the frauds

mappings = som.win_map(X)
# get co-ordinates from map
frauds = np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)
# revers scaling
frauds = sc.inverse_transform(frauds)

# analyst wil analyze
    

