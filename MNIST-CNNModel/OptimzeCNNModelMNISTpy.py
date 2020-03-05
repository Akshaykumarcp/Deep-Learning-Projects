# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:40:16 2020

@author: APadashetti
"""

# create CNN model and optimize it using keras tuner

# https://keras-team.github.io/keras-tuner/


import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images,test_labels) = fashion_mnist.load_data()

# scale down from 0 to 1 coz images are in gray scale

train_images = train_images / 255.0

test_images = test_images / 255.0 

train_images[0]

train_images[0].shape

# when 28 X 28 , resize into something which will be helpfull for CNN model. 
# in input layer we need to give how many images we have and what is the pixels along with the 4 D

train_images = train_images.reshape(len(train_images),28,28,1) 

test_images = test_images.reshape(len(test_images),28,28,1) 

# trying to optimize  CNN model to understand how many CNN layer have to take such as dense or flatten layers

# create a function for building model

def build_model(hp):  # hp = hyper parameter
  model = keras.Sequential([ # creating CNN model
    keras.layers.Conv2D( # add convolutional 2D layer. first parameter is filter
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)
        # hp.Choice = select values from multiple
        # when running keras tuner this will try min_valu and max_val what is the best value
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(), # flatten Conv2d layer 
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
            # in dense layer we can select how many no of nodes we can use between 32 -128. lot of permu and combi to select
    keras.layers.Dense(10, activation='softmax')
    # the last dense layer with 10 output nodes
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model

from kerastuner import RandomSearch # random search will try to find out which parameter will be best for problem statement
from kerastuner.engine.hyperparameters import HyperParameters

# run random search
# which conv layer and how many filters need to use

tuner_search = RandomSearch(build_model,objective='val_accuracy',max_trials=5,directory='CatsVsDogs',project_name='Mnist Fashion')

# will search for best parameters from the build_model
tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1) # will run only for 2 epochs by default. similar to fit 

model = tuner_search.get_best_models(num_models=1)[0]

model.summary()

# considering the above model, we'll train the images

model.fit(train_images,train_labels,epochs=10,validation_split=0.1,initial_epoch=3)