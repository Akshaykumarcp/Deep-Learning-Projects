# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:09:40 2020

@author: Akshay kumar C P
"""

# meaga case study - mke a hybrid DL model

# in the end we get ranking wit prob of fraud (using SOM we get less ranking)

# part 1 - identify the frauds with SOM

# copied below code from som.py

# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# frauds are those nearest to white and white. its based on our threshold
# (mappings[(5,3)], mappings[(8,3)] are the nearest to white and white co-ordinates

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# to convert from unsup to sup , we need dependent variable coz model need to find corelation bewtwen dependent and independent variable

# part 2 - going from unsupersived to supervised DL

# creating the matrix of feature

customers = dataset.iloc[:,1:].values

# creating rhe dependent variable (fraud, 1-f, 0-no fraud)

# frst we need - dep variable that contain fraud or not (0 and 1)

# initialize zero's first. frauds variable has fraudent ID. we take this ID and mark the respective rows as 1

is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Now let's make the ANN!

# data is very for DL model

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers,is_fraud, batch_size = 1, epochs = 2)

# Predicting the probailibties of frauds
y_pred = classifier.predict(customers)

# add cust ID 

# y_pred is 2d  so customers shud be 2d so convert for supporting concat

y_pred = np.concatenate((dataset.iloc[:,0:1].values,y_pred), axis = 1)

# sort based on prob

y_pred = y_pred[y_pred[:, 1].argsort()]
