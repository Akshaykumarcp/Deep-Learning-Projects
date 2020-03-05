# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 09:23:53 2020

@author: APadashetti
"""

# data augmentation using python and keras

# https://www.youtube.com/redirect?event=video_description&v=hxLU32zhze0&redir_token=zGh7tkbEI-XeQEESNDx0WYuniiB8MTU4MzQ2Njc1NEAxNTgzMzgwMzU0&q=https%3A%2F%2Fblog.keras.io%2Fbuilding-powerful-image-classification-models-using-very-little-data.html


# imagedatagenerator -  helps to generator to applly DA propertires such as flip,  shift , rotaion, range, zoom etc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range= 0.2,
        height_shift_range= 0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

img = load_img('14.jpg') # load image

x = img_to_array(img) # is a numpy array with shape (3,150,150)

x = x.reshape((1,) + x.shape) # is a numpy arrau wth shape (1,3,150,150). 3 channels - RBG. 150 X 150 pixels. 4 D 

# .flow() comand below generates bahces of randomly transformed images
# and saves the results to the 'preview/' directory

i = 0 
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',save_prefix='cat',save_format='jpeg'):
    i +=1 
    if i>20:
        break 