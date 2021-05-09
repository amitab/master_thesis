import tensorflow as tf
import numpy as np

from analyse_model_weights import analyse_model_weights
from analyse_models import analyse_models, analyse_models_v2, analyse_models_v2_and_dedup
from analyse_weights import analyse_weights


# mnet = tf.keras.applications.MobileNet()
# mnetv2 = tf.keras.applications.MobileNetV2()
vgg16 = tf.keras.applications.VGG16()
vgg19 = tf.keras.applications.VGG19()

# print("Starting to read files")
# w1 = np.loadtxt("weights/vgg19_-3.np")
# print("Read file 1")
# w2 = np.loadtxt("weights/vgg16_-3.np")
# print("Read file 2")

# print(
#     analyse_weights(
#         vgg16.layers[-3].weights[0].numpy(), vgg19.layers[-3].weights[0].numpy(),
#         {
#             'fp': [0.01, 0.001], # for different floating point thresholds
#             'sim': [.7, .8, .9], # for naive diff similarity percentage
#             # 'diff': [.1, .2, .3], # for lsh difference
#         },
#         500, 500 # block dims
#         # , 512 # bits
#     )
# )

# print(
#     analyse_model_weights(
#         mnet,
#         mnetv2,
#         {
#             'fp': [0.01, 0.001],  # for different floating point thresholds
#         },
#         32
#     ))

data = \
    analyse_models_v2_and_dedup(
        vgg16,
        vgg19,
        {
            'fp': [0.01, 0.001],  # for different floating point thresholds
            'sim': [.7, .8, .9],  # for naive diff similarity percentage
            # 'diff': [.1, .2, .3], # for lsh difference
        },
        500,
        500,
        32
        # , 256
    )

import pdb
pdb.set_trace()