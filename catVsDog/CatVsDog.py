# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 05:47:26 2020

@author: APadashetti
"""

# CNN
# Part 1

# import lib's

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense 

# iniialize the CNN

classifier = Sequential()

# add different layers 

# step -1 convolution (image -> feature detector --> feature map -- al together convlayer)
# 32 - comomnly selected filter. 3 X 3 is the size of filter
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu"))

# step 2 - pooling,  maxpool to reduce all cretirea
# apply maxpool to all feature map 

classifier.add(MaxPooling2D(pool_size = (2,2)))

# additing 1 more conv layer for improving validation/test set accuracy to overcome overfitting
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))


# step 3 - flatening .
# pooled feature map into single vector

classifier.add(Flatten())

# step 4 - fully connectted layer 
# add hidden layer (fully connecte layer), add output layer
# no thumb rule to add nodes in hidden layer but common practice is 128 .

classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 1,activation = 'sigmoid'))

# compile using stocastic , acuracy etc

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

# image preprocesing

# Part -2 fitting the CNN to the mages

# image augumentation - keras - before fitting into CNN. overcome overfitting

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# part - 3 making new predictions

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64, 64))

# add new dim to test_image coz CNN in input_shape has 64,64,3. 3 is the colored image so need to use img to arry for dim

test_image = image.img_to_array(test_image) # from 2d 64 X 64 2D to  64 X 64 X 3

# add 1 dim again coz predict method needs 4 dimensions. Basically need to 1 dim to test_image just to give for predict. functions lik predict acceptsinpute as batch . 
# 1 bathch 1 input here. 

test_image = np.expand_dims(test_image,axis=0) # gives 4D , 1,64,64,3

result = classifier.predict(test_image) 

# don't whether1 is cat or dog so

train_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


# improve validation test accuracy
    
# https://www.udemy.com/course/deeplearning/learn/lecture/7023358#questions/2276518
