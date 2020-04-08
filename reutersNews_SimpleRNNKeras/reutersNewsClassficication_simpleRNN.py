# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 04:36:19 2020

@author: Akshay kumar C P
"""
# conda create -n newEnv

# activate newEnv

# PIP INSTALL TENSORFLOW==1.14 keras

# tensorflow - 1.1.4

# classification problem

# simple RNN (stacked vanila)

import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import reuters # dataset
from keras.preprocessing.sequence import pad_sequences # performig padding , length of each row is not equal so to make equal we add zero's 
from keras.utils import to_categorical # categorical operation, i,e classify
from keras.models import Sequential # sequential model, side by side. 
from keras.layers import Dense, SimpleRNN, Activation # dense - outer layer. activation - activation function
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier # coz clasification

# parameters for data load

num_words = 30000  # to load 
maxlen = 50 # of word/line
test_split = 0.3 

(X_train,y_train),(X_test,y_test) = reuters.load_data(num_words=num_words,maxlen = maxlen,test_split=test_split )

# pad sequences with zeros
# padding parameter s set to post = 0's are appenedded to end of sequences

X_train = pad_sequences(X_train,padding = 'post') # entire line
X_test = pad_sequences(X_test,padding = 'post') #classes, predictng for. 

# few lines will be les than 50, so after ex 25 length add zeros = post padding 

X_train = np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1)) #
X_test = np.array(X_test).reshape((X_test.shape[0],X_test.shape[1],1))

y_data = np.concatenate((y_train,y_test))
y_data = to_categorical(y_data)
y_train = y_data[:1395]
y_test = y_data[1395:]  

# in vanila rnn only 
def stacked_vanila_rnn():
    model = Sequential() # creates pipline
    model.add(SimpleRNN(50,input_shape=(49,1),return_sequences=True)) # 50 neuron per layer. r_seq = provide output to next layer
    model.add(SimpleRNN(50,return_sequences = False))
    model.add(Dense(46))
    model.add(Activation('softmax')) # softmax will find prob of eacha nd every class of dataset
    
    adam = optimizers.Adam(lr=0.001) # generally 0.3 - 0.001
    model.compile(loss = 'categorical_crossentropy', optimizer =adam,metrics=['accuracy'])
    
    return model

model = KerasClassifier(build_fn = stacked_vanila_rnn,epochs = 2, batch_size =50, verbose=1) # epch will be 30k , 40k. just for demo less . verbose = 1 -- print stack / model
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test,axis=1)

print(accuracy_score(y_pred,y_test))

# require tensorflow 1.X cersion. so error's while running

# increase epoch, layers, optimizers different etc for more accu