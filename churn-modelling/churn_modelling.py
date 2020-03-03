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

print(CM)

from sklearn.metrics import accuracy_score

acc_score = accuracy_score(y_pred,y_test)

print(acc_score)

# training , test and validation accuracy are similar i,e perfect model

# next video
# hpyerparamter tuning, how many hiden neurons and layers to be used in ANN. General way we'll c while creating ANN

# perform hyperparameter optimization

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization
from keras.activations import relu, sigmoid

'''
def create_model(layers, activation): # layrs and type of activation
    model = Sequential() # initiaze sequential model
    for i, nodes in enumerate(layers):
        # first layer
        if i==0: 
            model.add(Dense(nodes,input_dim=X_train.shape[1])) # first layer providing input dimention 
            model.add(Activation(activation)) # specifies which type of activation needed
            model.add(Dropout(0.3)) # applying dropout rate
        else:
            # creating all the hidden layers
            model.add(Dense(nodes))
            model.add(Activation(activation)) # apply activ
            model.add(Dropout(0.3))
            
            #last layer. units = output = 1. based on research uniform works well
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

# build_fn = created function
model = KerasClassifier(build_fn=create_model,verbose=0)

# in NN 20, 1 HL 20 Neuron.
layers =[[20],[40,20],[45,30,15]]
activations = ['sigmoid','relu'] # example : first iteration sigmoid, second itration rely for 20
param_grid = dict(layers=layers,activation=activations,batch_size = [128,256],epochs=[30])
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=5) # cv of 5 experiments

grid_result = grid.fit(X_train,y_train)
'''

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)


layers = [[20], [40, 20], [45, 30, 15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(X_train, y_train)

'''
[0.8387499995157123,
 {'activation': 'relu',
  'batch_size': 256,
  'epochs': 30,
  'layers': [45, 30, 15]}]
'''

print(grid_result.best_score_,grid_result.best_params_)


pred_y = grid.predict(X_test)

y_pred = (pred_y > 0.5)

y_pred

'''

array([[False],
       [False],
       [False],
       ...,
       [False],
       [False],
       [False]])
'''


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)

score

