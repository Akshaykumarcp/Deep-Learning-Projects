# TEST GPU SET UP via commands

# run commands below:
# 1. nvcc -V --> nvidia cuda compiler driver
# 2. nvidia-smi --> about GPU

# cudNN version can be found at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include\cudnn.h as below:

#define CUDNN_MAJOR 7
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 5

# TEST GPU SET UP via tendorflow

import tensorflow as tf
from tensorflow.python.client import device_lib

# list available tensorflow devices
print(device_lib.list_local_devices())

# check tf version
tf.__version__

# is cuda installed ?
tf.test.is_built_with_cuda()

# test whether GPU is available
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# physical_device name
tf.config.list_physical_devices('GPU')

# number of GPU's available
len(tf.config.experimental.list_physical_devices('GPU'))

# code to confirm tensorflow is using GPU
tf.config.experimental.list_physical_devices('GPU')

# CONFIG GPU

from tensorflow._api.v2.compat.v1 import ConfigProto
from tensorflow._api.v2.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# avoid using 100% of GPU, else GPU overclock.
config = ConfigProto()
# use 50% of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# BEGIN THE PROGRAM

# import the lib's

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from glob import glob

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'dataset/train'
valid_path = 'dataset/test'

# use imagenet weights
inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# not to train existing weights
for layer in inception.layers:
    layer.trainable = False

# get no of output classes
total_no_of_classes = glob('dataset/train/*')

# flatten all the layers
x = Flatten()(inception.output)

output_layer = Dense(len(total_no_of_classes), activation='softmax')(x)

# create object for the model
inception_model = Model(inputs=inception.input, outputs=output_layer)

# show model architecture
inception_model.summary()

# inform model about cost and optimization method to use
inception_model.compile(
    loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])

# utilize 'Image Data Generator' for importing images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# give same target size as input size for the images
training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# run the inceptionV3 model
result = inception_model.fit(
  training_set,
  validation_data=test_set,
  epochs=25,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

'''
Epoch 20/20
574/574 [==============================] - 160s 278ms/step - loss: 1.3683 - accuracy: 0.9219 - val_loss: 3.5091 - val_accuracy: 0.8595
'''

import matplotlib.pyplot as plt

# plot the loss
plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(result.history['accuracy'], label='train acc')
plt.plot(result.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save resnet50 model to local
inception_model.save('models/inceptionV3_model.h5')

# prediction for testset
predict_new = inception_model.predict(test_set)

predict_new

import numpy as np
# take argmax on testset i,e on all images
predict_new = np.argmax(predict_new, axis=1)
predict_new


# load model from local
model=load_model('models/inceptionV3_model.h5')

# load new image
new_image = image.load_img('dataset/validation/test/Tomato___Early_blight/0f111718-a749-42cf-9495-412f4b51acf0___RS_Erly.B 7482.JPG',target_size=(224,224))
#H:\AI\DL-projects\tomato\dataset\validation\test\Tomato___Early_blight\0f111718-a749-42cf-9495-412f4b51acf0___RS_Erly.B 7482.JPG
# convert PIL image type to array
new_image_array = image.img_to_array(new_image)

new_image_array.shape

# normalize
new_image_array=new_image_array/255

import numpy as np
# add additional dim for input of the NN
new_image_array=np.expand_dims(new_image_array,axis=0)
new_image_array =preprocess_input(new_image_array)
new_image_array.shape

# predict new image
model.predict(new_image_array)

# argmax on predicted image
maxVoter=np.argmax(model.predict(new_image_array), axis=1)

maxVoter==2
# true

'''
Error's come accrossed:

1. ImportError: Could not import PIL.Image. The use of `load_img` requires PIL.
solution: pip install --upgrade keras numpy pandas sklearn pillow (https://github.com/asataniAIR/Image_DL_Tutorial/issues/4)

2. Internal: Invoking GPU asm compilation is supported on Cuda non-Windows platforms onlyRelying on driver to perform ptx compilation. Modify $PATH to customize ptxas location.
ref: https://github.com/tensorflow/models/issues/7640

3. could not synchronize on CUDA context: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered :: 0x00007FFE2B93BA05	tensorflow::CurrentStackTrace. GPU sync failed
solution: restart the program. (https://stackoverflow.com/questions/51112126/gpu-sync-failed-while-using-tensorflow)
'''



