# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 07:01:36 2020

@author: Akshay kumar C P
"""


import numpy as np  # arrays
import pandas as pd # for datasets
import torch
import torch.nn as nn # nn is the module of torch for implement NN that has all necessary things for AE
import torch.nn.parallel #parelel computation
import torch.optim as optim # for potimizer
import torch.utils.data
from torch.autograd import Variable # stocastic GD

# import dataset
# encoding - coz movie name contain special character
movies = pd.read_csv('ml-100k/ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
# not using movies to train and test set, just to show the movies.
# same for with user and ratings

users = pd.read_csv('ml-100k/ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')

ratings = pd.read_csv('ml-100k/ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

# preparing training and test set
# from 1000k data set folder, devided users into 5-fold data where base is training set and test is test data. 
# ususally 10-CV, here data is ready for 5-fold.
# train for training 5-fold. test for testing after 5-fold.
# here taking only u1.base for learning AE and not doing k-fold CV

training_set = pd.read_csv('ml-100k/ml-100k/u1.base',delimiter = '\t')
# to aray convert , coz in python 
training_set = np.array(training_set,dtype='int')


test_set = pd.read_csv('ml-100k/ml-100k/u1.test',delimiter = '\t')
# to aray convert , coz in python 
test_set = np.array(test_set,dtype='int')

# getting the no of users and movies
# next we gonna convert training_set and test_set to a matrix where columns - movies , lines - users, sales  = ratings. for trainingset and testset. 
# for each matrix , we include all user and movers. if user dint rate movie then 0 

nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# converting rhe data into a array with users in line and movies in columns

# y to convert into matrix as lines as users and columns as movies
# coz need make a specific structure of data that'll corespond to RBM excepts as inout
# RBM are type of NN wher ipute nodes , there are features. there are observatiosn going 1 by 1 into a nw starting from input node
# we need to create a structure that'll contain these observation and thier differnent feature that are going to indded in nodes

# lets convert
# both traininset and testset

def convert(data):
    # create list od list rather than 2D
    new_data = []
    for id_users in range(1,nb_users + 1):
        id_movies = data[:,1] [data[:,0]== id_users]
        id_ratings = data[:,2] [data[:,0]== id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# converting the data into torch tensors

# usual structure for DL model
# we'll start creating architure of NN
# using pytorch tensors
# tensors - are arrays that contains elements of signle datatype. pytorch array.

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Data preprocessing is common until here for RBM and autoencoders

# class building, define RBM shud be built . no of hidden nodes, thats the weights, prob of visible nodes given the hiden node, bais for same probl are the first parameter need for RBM

# AE

# create child class

# stacked AE coz we've got many hidden layers , several encodings 
#inheritance 

class SAE(nn.Module):
    #init used to inialize the class via object
    def __init__(self,):
        # super coz to use methods and classes from nn modules
        super(SAE,self).__init__()
        # start creat arch of NN - no of layeras and neurons in each layer
        # 1st part of NN i,e full connection between the input vector and 1st HL 
        #first full connection - fcl. try several values for neurons in HL - here 20 nodes in first HL
        # 20 Hidden neuron represents feature extraction through unsupervised detects.  features such as movies liked by similar people. 
        # one of the feature could be gener horror, when new user come to watch horror movie - if this user gave good ratings to movie then 
        # then horor genre neuron is activation and therefor big wieght will be attributed to this neiron in the final prediction.
        
        self.fc1 = nn.Linear(nb_movies,20)
        self.fc2 = nn.Linear(20,10)
        # 10 will detect wven more features but based on 20.
        # encoding is done
        # here starting to decode, reconstruct input vector. let's make symetrical i,e 20 
        self.fc3 = nn.Linear(10,20)
        self.fc4 = nn.Linear(20,nb_movies)
        
        # that's the arch for AE
        
        # specify activation 
        # tried rectifier and sigmoid - sigmoid was good 
        self.activation = nn.Sigmoid()
        
    # next function to actions of encoding and decoding
    # and activation for full conections
    # returns the vector of predicted ratnngs
    
    def forward(self,x): #x input vector 
        x = self.activation(self.fc1(x)) # encoding
        x = self.activation(self.fc2(x)) # encoding
        x = self.activation(self.fc3(x)) # encoding
        #decoding
        x = self.fc4(x)
        return x
        
sae = SAE() # AE is ready

criterion = nn.MSELoss()

# optimizer 

optimizer = optim.RMSprop(sae.parameters(),lr=0.01,weight_decay=0.5)

# training SAE

nb_epoch = 200
# 1 loop for epoch and 1 loop for all user's

for epoch in range(1,nb_epoch + 1):
    train_loss = 0
    s = 0. # no of user atleast rated 11 movie. float
    for id_user in range(nb_users):
        # action for each user
        input = Variable(training_set[id_user]).unsqueeze(0) # tensore doesn;t take 1D as inpout so vairable and create batch 
        # target = clone of input
        target = input.clone()
        # if user has not rated single movie then ignore
        if torch.sum(target.data > 0)>0 :
            output = sae(input)
            target.require_grad = False # not to cal gradient for target has it is a clone
            output[target==0] = 0 # these values doesn't calcu in computation of error. after error wieghts are updated so save computation
            # compute loss error
            loss = criterion(output,target)
            mean_coreector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # for movies non-xero.  # if denominator is zero , problem so 1e
            
            # next - backward method for loss , tells in which direction nedd to update weights , do v need to increase/decrease weights
            loss.backward()
            
            train_loss += np.sqrt(loss.item()*mean_coreector)
            s += 1.
            
            # use optimizer
            optimizer.step() # to update weights, amount of which updates weights
            
    # print for each epoch
    print('epoch:'+str(epoch)+'loss:'+str(train_loss/s))
            
            
# loos is average loss betwen real and predicted value
# doing on train set


# epoch:200loss:0.9118096904725974

# shud also do on test set 

# testing the SAE

test_loss = 0
s = 0. # no of user atleast rated 11 movie. float
for id_user in range(nb_users):
        # action for each user
    input = Variable(training_set[id_user]).unsqueeze(0) # tensore doesn;t take 1D as inpout so vairable and create batch 
        # target = clone of input
    target = Variable(test_set[id_user])
        # if user has not rated single movie then ignore
    if torch.sum(target.data > 0)>0 :
        output = sae(input)
        target.require_grad = False # not to cal gradient for target has it is a clone
        output[(target==0).unsqueeze(0)] = 0 # tensore doesn;t take 1D as inpout so vairable and create batch 
        # The torch.Tensor returned by target == 0 is of the shape [1682].
        #(target == 0).unsqueeze(0) will convert it to [1, 1682]
        # target = clone of input] = 0 # these values doesn't calcu in computation of error. after error wieghts are updated so save computation
        # compute loss error
        loss = criterion(output,target)
        mean_coreector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # for movies non-xero.  # if denominator is zero , problem so 1e
        test_loss += np.sqrt(loss.item()*mean_coreector)
        s += 1.
            
    # print for each epoch
print('test loss:'+str(test_loss/s))

# test loss:0.9499882644313711
       