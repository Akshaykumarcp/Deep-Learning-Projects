# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 07:15:10 2020

@author: Akshay kumar C P
"""

'''
problem statement :
    
    drug vaccination between age 13-100 years
    
    people > 65 --> observed some side effects -- 1
    people < 65 no side effect -- 0

'''

import numpy as np # array to which input for keras model
from random import randint
from sklearn.preprocessing import MinMaxScaler

# data collection
train_sample = [] # features
train_label = [] # 0 or 1 

for i in range(1000):
    younger_ages = randint(13,64)
    train_sample.append(younger_ages)
    train_label.append(0)
    
    older_ages = randint(65,100)
    train_sample.append(older_ages) 
    train_label.append(1)
    
    
test_sample = [] # features
test_label = [] # 0 or 1 

for i in range(500):
    younger_ages = randint(13,64)
    test_sample.append(younger_ages)
    test_label.append(0)
    
    older_ages = randint(65,100)
    test_sample.append(older_ages) 
    test_label.append(1)
    
    
# data collection completed

# next convert into numpy array as keras expects input as array
    
train_sample = np.array(train_sample)
train_label = np.array(train_label)

test_sample = np.array(test_sample)
test_label = np.array(test_label)


# no varies in train_sample , so let's scalre between 0 to 1

scalar = MinMaxScaler(feature_range=(0,1))
# reshape(-1,1) coz train_sample is 1D so need to reshape as keras model understands
scalar_train_sample = scalar.fit_transform(train_sample.reshape(-1,1))

# create ANN 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 2 HL. no of neurons in first HL is 16
model = Sequential([Dense(16,input_dim=1,activation='relu'),Dense(32,activation='relu'),Dense(2,activation='softmax')]) 

model.summary()

# after compile wirghts in the NN gets changed
model.compile(Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_sample,train_label,batch_size=10,epochs=10)

# model is ready for predicting

test_sample_predicted = model.predict_classes(test_sample,batch_size=10)

# CM

from sklearn.metrics import confusion_matrix

test_sample_predicted_CM = confusion_matrix(test_label,test_sample_predicted)