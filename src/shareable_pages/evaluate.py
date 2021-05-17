import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.datasets import mnist
import keras

# from analyse_model_weights import analyse_model_weights
# from analyse_models import analyse_models, analyse_models_v2, analyse_models_v2_and_dedup
# from analyse_weights import analyse_weights


# DATA_DRIFT_MODELS_PATH = "../drift/Concept Drift (Data)/"

# m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + 'based_model/based_model-45')
# m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + '30k_normal_added_10k_mix/30k_normal_added_10k_mix-45')

# m1._name = "based_model"
# m2._name = "30k_normal_added_10k_mix"

# sizes = [i * 10 for i in range(1, 51)]

# for size in sizes:
#     print(
#         analyse_models_v2_and_dedup(
#             m1, m2,
#             {
#                 'fp': [0.01, 0.001],  # for different floating point thresholds
#                 'sim': [.7, .8, .9],  # for naive diff similarity percentage
#                 # 'diff': [.1, .2, .3], # for lsh difference
#             },
#             size,
#             size,
#             "./",
#             0.01,
#             build_dedup=True
#         ))


# Generate the train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
 
# Generate some noisy data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

import os
import gc

fiel = open('based_model_30k_normal_added_10k_mix_10_500', 'w+')

for name in os.listdir("./models"):
    if 'based_model' in name or '30k_normal_added_10k_mix' in name:
        fiel.write(f"{name}\n")
        data = name.rsplit("_", 5)
        block_size = (int(data[-2]), int(data[-1]),)
        fp_threshold = float(data[1])
        sim_threshold = float(data[2])
        weight_lower_bound = float(data[3])
        model_name = data[0]
    
        fiel.write(f"Model: {model_name}, weight >= {weight_lower_bound}MB, block size: {block_size[0]} x {block_size[1]}, fp threshold: {fp_threshold}, sim threshold: {sim_threshold}\n")

        model = tf.keras.models.load_model(f"./models/{name}")

        results = model.evaluate(x_test, y_test)
        fiel.write(f"Accuracy - MNIST: {results}\n")
        results = model.evaluate(x_test_noisy, y_test)
        fiel.write(f"Accuracy - MNSIT Noisy: {results}\n")
        fiel.write("-----------------------------------------------------------------------------------------------------------------------\n")

        fiel.flush()

        tf.keras.backend.clear_session()
        del model
        gc.collect()