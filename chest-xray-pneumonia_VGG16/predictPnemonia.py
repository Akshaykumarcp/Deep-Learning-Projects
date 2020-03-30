# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:14:17 2020

@author: Akshay kumar C P
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np
model = load_model('modelForPnemonia_vgg16.h5')

img = image.load_img('chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(224,224))

x = image.img_to_array(img)

x = np.expand_dims(x,axis=0)

img_data = preprocess_input(x)

classes = model.predict(img_data)