# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 05:20:21 2020

@author: Akshay kumar C P
"""
# Recurrent neural network

# implement LSTM - powerfull model 
# robust etc

# part 1 : data preprocessing

# import lib's

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import training set
# make  numpy array because only arrys can be input to tthe NN and keras
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:,1:2] # getting only 1st index column. 2 is just the upper bound

# feature scalling
# 1. standardization and 2. normalizatoion

# when sigmoid is used in the output layer of RNN better to use normalization
# MinMaxscaler class

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)

# fit - get min of the data and max of the data
# transform - compute 

# creating a data structure with 60 timesteps and 1 output 

# 60 timesteps - at each t, the Rnn will lokk at the 60 stocks befor time t and based on trends it'll predict next t + 1
# 60 is the no of times experimented. 1 time steps - model  was not learning anything. 10, 20 ,30 ,40 model got learning trends so ended up with 50

# 1st thing create 2 entity , xtrain - input to model(60 previwes) . ytrain (60 next previews) - output 

X_train = [] # empty list
y_train = []

# 60 in xtrain and next 60 in ytrain

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0]) # 0 th index 
    y_train.append(training_set_scaled[i,0]) # 60 + 
    # make  numpy array because only arrys can be input to tthe NN and keras
X_train,y_train = np.array(X_train),np.array(y_train)

# last step 

# reshaping  (add dim i,e unit (no of predictors) predictors are indicator. now indicator we have is 0th index i,e open stock)

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) # currently in 2D , reshaoe to 3D. for giving dim fo to keras doc - recurent layers - input shapes


# part 2 - building the RNN  (not simple RNN. stacked LSTM with dropout regularization for preventing overrfitting)

# import keras lib's

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialize the RNN

regressor = Sequential() # continous values so regressor
# build differnet layer

# add 1st LSTM layer and some dropout regularization (overcome overfitting while predicting)
# units = neurons . 50 will get a model with high dim. if 3-5 neurons will niot capture low and high trends

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1) )) # 3 arguments. 1st = no of units which is no of units sell . relevant no. 2nd = return sequences = true , 3rd = input shape . shape of X_train
regressor.add(Dropout(0.2)) # classic no is to drop 20 % of neuron will be ignored in forward and backpropogation

# add 2st LSTM layer and some dropout regularization (overcome overfitting while predicting)

regressor.add(LSTM(units = 50, return_sequences = True)) # 3 arguments. 1st = no of units which is no of units sell . relevant no. 2nd = return sequences = true , 3rd = input shape . shape of X_train
regressor.add(Dropout(0.2)) # classic no is to drop 20 % of neuron will be ignored in forward and backpropogation


# add 3st LSTM layer and some dropout regularization (overcome overfitting while predicting)

regressor.add(LSTM(units = 50, return_sequences = True)) # 3 arguments. 1st = no of units which is no of units sell . relevant no. 2nd = return sequences = true , 3rd = input shape . shape of X_train
regressor.add(Dropout(0.2)) # classic no is to drop 20 % of neuron will be ignored in forward and backpropogation

# add 4st LSTM layer and some dropout regularization (overcome overfitting while predicting)

regressor.add(LSTM(units = 50)) # 3 arguments. 1st = no of units which is no of units sell . relevant no. 2nd = return sequences = true , 3rd = input shape . shape of X_train
regressor.add(Dropout(0.2)) # classic no is to drop 20 % of neuron will be ignored in forward and backpropogation

# add final layer i,e output layer (last step of arch of NN)

regressor.add(Dense(units=1))

# 2 steps remainig . 
# 1. compile th e RNN with powerfull optimizer and loss (MSE)
# 2. fit to training set

# compile the RNN
# refer keras doc - optimizer 
regressor.compile(optimizer='adam',loss = 'mean_squared_error')

# fit the RNN to the training set

regressor.fit(X_train,y_train,epochs=100,batch_size=32)

# dont decrease the loss to complete xero coz may get overfit

#part 3 - making the prediction and visualising the results

# geting the real stock price of 2017

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2] # getting only 1st index column. 2 is just the upper bound

# getting the predicted stock of 2017
# important tricks
# key point 1 = we trained the model to be able to predict stock time t+1 based on 60 previews stock price therefor to predict each fianancial day 2017 we need 60 previews of stock prices of 60 previous financial days before the actual day
# 2 - to get previous 60 days , we need both train adn test.sum of the 60 will train dec 2016 and some of them will come from jan 2017 test set. concat both

# 3 - scale the training and test set. problem is we'll change actual test values. shud not do lik. keep actual test values. so concate differently i,e concat raw train and test, concat them and then scale so that we preserve the actual values from test set

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
# for each day of jan 2017 need 60 previous stock prices. There are 20 financial days in a month, therfore 60 days would make 3 months. 

inputs = dataset_total[len(dataset_total) - len(dataset_test)-60 : ].values # .values to arrays

inputs = inputs.reshape(-1,1)
# expected from NN need to have proper format - 3D
# before that scale the inputs 
# here not using fitting because same scalling shud be applied as above 

inputs = sc.transform(inputs)

X_test = []
# 60 + 20 (predicting)
for i in range(60,80):
    X_test.append(inputs[i-60:i,0]) # 0 th index 
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1)) # currently in 2D , reshaoe to 3D. for giving dim fo to keras doc - recurent layers - input shapes

predicted_stock_price = regressor.predict(X_test)

# inverse the scaleing out prediction , to get original values back 

predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# visualising the results

plt.plot(real_stock_price,color = 'red',label = 'real google stock price')

plt.plot(predicted_stock_price,color = 'blue',label = 'predicted google stock price')

plt.title("Google stock price prediction")
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

'''
evaluating the model with the RMSE does not make much sense, since we are more interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price. We want to check if our predictions follow the same directions as the real stock price and we don’t really care whether our predictions are close the real stock price. The predictions could indeed be close but often taking the opposite direction from the real stock price.

Nevertheless if you are interested in the code that computes the RMSE for our Stock Price Prediction problem, please find it just below:

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
Then consider dividing this RMSE by the range of the Google Stock Price values of January 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is more relevant since for example if you get an RMSE of 50, then this error would be very big if the stock price values ranged around 100, but it would be very small if the stock price values ranged around 10000.

'''

'''
here are different ways to improve the RNN model:

Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.
'''

'''
you can do some Parameter Tuning on the RNN model we implemented.

Remember, this time we are dealing with a Regression problem because we predict a continuous outcome (the Google Stock Price).

Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:

scoring = 'accuracy'  

by:

scoring = 'neg_mean_squared_error' 

in the GridSearchCV class parameters.
'''