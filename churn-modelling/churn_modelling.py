# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 08:20:25 2020

@author: Akshay kumar C P
"""

'bank customer stay or not'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13] 

y = dataset.iloc[:,13]

# create dummy variables

geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

X = X.drop(['Geography','Gender'],axis=1)

X = pd.concat([X,geography,gender],axis=1)

# split dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# feature scaling

# need to scale because multplcation and backpropogation and GD happens well
from sklearn.preprocessing import StandardScaler

SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)

# next, Let's create a ANN 

# import keras replated lib's

from keras.models import Sequential # every model NN , CNN, RNN. responsible for creating NN
from keras.layers import Dense # for create hidden layer's
# from keras.layers import LeakyReLU, PReLU, ELU # activation function
from keras.layers import Dropout #dropout layer, regularization parameter

# initialising the ANN
classifer =  Sequential() # empty NN

# adding the input layer and first Hiden layer
# output dim (units) = 6 hiiden neuron, innit (kernel_initializer) = how weight need to be intialized. he_uniform and he_normal works well for relu.
# input_dim = 11, how many input features are present X_train = 11 features
classifer.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation = 'relu',input_dim=11))

# apply drop out when deep NN
#classifer.add(Dropout(0.3))

# adding the second hidden layer
classifer.add(Dense(units = 6,kernel_initializer = 'he_uniform',activation = 'relu'))
#classifer.add(Dropout(0.4))
# at hidden layer relu is good, for output layer's sigmoid is good 

# adding the output layer
# units = 1 because binary classfication. if > 0.5 answer will be 1
classifer.add(Dense(units = 1,kernel_initializer = 'glorot_uniform',activation = 'sigmoid'))

# details about NN
classifer.summary()

# compile ANN

classifer.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model_hist = classifer.fit(X_train,y_train,validation_split=0.33,batch_size=10,nb_epoch = 100)

# simple NN and right initializer (he_uniform) accuracy Epoch 100/100
# 5359/5359 [==============================] - 9s 2ms/step - loss: 0.3251 - accuracy: 0.8649 - val_loss: 0.3572 - val_accuracy: 0.8554

# list all data in history

print(model_hist.history.keys())

'''
Making the prediction and evaliatiing the model
'''

# predicting test set results

y_pred = classifer.predict(X_test)

y_pred = (y_pred > 0.5)

# making the confusion matrix

from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test,y_pred)

# calculate the accuracy

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_pred,y_test)

# training , test and validation accuracy are similar i,e perfect model

