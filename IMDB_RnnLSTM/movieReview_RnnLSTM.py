# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 06:13:06 2020

@author: Akshay kumar C P
"""

# moview review 0 or 1 classification

# data already in vector transformation so noneed to vector transform
# if dataset is taken from net, need to vector transform

from __future__ import print_function

from sklearn.metrics import accuracy_score
from keras.datasets import imdb # dataset
from keras.preprocessing.sequence import pad_sequences # performig padding , length of each row is not equal so to make equal we add zero's 
from keras.models import Sequential # sequential model, side by side. 
from keras.layers import Dense, Embedding, LSTM # dense - outer layer. activation - activation function
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier # coz clasification

# parameters for data load

max_features = 20000  # no of features
maxlen = 80 # only take 80 len of review. cut text after this no of words (among top max features most common words)
batch_size = 36 

print("loading data..")
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=max_features)

print(len(X_train),'train sequences')
print(len(X_test),'test sequences')

# pad sequences with zeros
# padding parameter s set to post = 0's are appenedded to end of sequences

print('pad sequences (samples x times')
# by defualt pre padding
X_train = pad_sequences(X_train,maxlen=maxlen) # entire line
X_test = pad_sequences(X_test,maxlen=maxlen) #classes, predictng for. 

print(X_train.shape,'xtrain shapre after padding')
print(X_test.shape,'xtest shapre after padding')

print('build model')

model = Sequential() # creates pipline
# embeding values into a network. turns +ve intergers into a dense vectors of fixed sizes. used as a first layer in a model
model.add(Embedding(max_features,128)) # prepares a input dataset
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2)) # add lstm layers. 128 lstm. by default other ateributes are ther. dropout = sigmoid . recurent_drop out is all the sigmoid in LSTM
model.add(Dense(1,activation='sigmoid'))

# try using dif optimizers and dif optimize config

model.compile(loss = 'binary_crossentropy', optimizer ='adam',metrics=['accuracy'])
    
print('train..')

model.fit(X_train,y_train,
          batch_size = batch_size,
          epochs=1,
          validation_data=(X_test,y_test)
          )

model.predict(X_test)

score, acc = model.evaluate(X_test,y_test,
                            batch_size= batch_size )

print('test score:', score)
print('test acc:', acc)

model.predict(X_test)

'''
Out[2]: 
array([[0.12336606],
       [0.8876357 ],
       [0.5647729 ],
       ...,
       [0.12664345],
       [0.03767127],
       [0.8205288 ]], dtype=float32)
'''


# increase epoch, batch size, diff epochs etc for more accu