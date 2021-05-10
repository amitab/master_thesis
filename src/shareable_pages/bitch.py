import os
import math
import numpy as np
import cv2 as cv
import keras
import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow.keras import applications
import tensorflow.keras.applications.vgg16 as vgg16

from keras import optimizers
from keras.preprocessing import image
from keras.utils import to_categorical

# datasets, info = tfds.load(name="imagenet2012_subset", 
#                         with_info=True, 
#                         as_supervised=True, 
#                         download=False, 
#                         data_dir="/run/media/thekeystroker/Overflow/datasets")

# C = info.features['label'].num_classes
# Ntrain = info.splits['train'].num_examples
# Nvalidation = info.splits['validation'].num_examples
# Nbatch = 64

# train_dataset, validation_dataset = datasets['train'], datasets['validation']


datasets, info = tfds.load(name="imagenet_a", 
                        with_info=True, 
                        as_supervised=True, 
                        download=True, 
                        data_dir="/run/media/thekeystroker/Overflow/datasets")

C = info.features['label'].num_classes
Nvalidation = info.splits['test'].num_examples
Nbatch = 64

validation_dataset = datasets['test']


def imagenet_generator(dataset, batch_size=Nbatch, num_classes=1000, is_training=False):
  images = np.zeros((batch_size, 224, 224, 3))
  labels = np.zeros((batch_size, 1))
  while True:
    count = 0 
    for sample in tfds.as_numpy(dataset):
      image = sample[0]
      label = sample[1]
    
      images[count%batch_size] = vgg16.preprocess_input(np.expand_dims(cv.resize(image, (224, 224)), 0))
    #   labels[count%batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)
      labels[count%batch_size] = label

      count += 1
      if (count%batch_size == 0):
        yield images, labels

model = vgg16.VGG16(weights='imagenet', include_top=True)

model.compile('sgd', 'categorical_crossentropy', ['accuracy', 'sparse_categorical_accuracy','sparse_top_k_categorical_accuracy'])


score = model.evaluate_generator(imagenet_generator(validation_dataset,batch_size=Nbatch), 
                                  steps= Nvalidation // Nbatch, 
                                  verbose=1)
print(score)