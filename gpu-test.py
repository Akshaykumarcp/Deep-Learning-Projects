# CREDITS: https://www.tensorflow.org/guide/gpu

import tensorflow as tf

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

