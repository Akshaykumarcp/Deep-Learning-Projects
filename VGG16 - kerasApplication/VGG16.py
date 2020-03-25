# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:21:08 2020

@author: Akshay kumar C P
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16 # transfer learning, imagenet has 1000 categories of trained output. in last layer we're using our req is caled as TL

# for other keras applications
# from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator # image augumentation
from keras.models import Sequential # sequential model
import numpy as np

from glob import glob
import matplotlib.pyplot as plt

# resize all the images to this
IMAGE_SIZE = [224,224]  # coz VGG16 image is 224 X 244 

train_path = 'Datasets/Train'
test_path = 'Datasets/Test'

# add preprocessing layer to the front of VGG
# include_top=False , removing last layer i,e 1000 cat
# ResNet50 for reset. and change fromm vgg to res everywhere
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet',include_top=False)


# dont train existing weights
# we don't ve to train ma existing VGG16 layers coz already train
for layer in vgg.layers:
    layer.trainable = False # trainable is a parameter
    
# usefull for geting no of classes
folders = glob('Datasets/Train/*')

# flatten last layer of VGG16
# can add more if we want
x = Flatten()(vgg.output)

prediction = Dense(len(folders),activation='softmax')(x)

# create a model object
model = Model(input=vgg.input, outputs=prediction)

# view the structure of model

model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures.h5')