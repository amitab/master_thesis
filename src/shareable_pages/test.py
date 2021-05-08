import tensorflow as tf
import numpy as np

from analyse_model_weights import analyse_model_weights
from analyse_models import analyse_models
from analyse_weights import analyse_weights


# mnet = tf.keras.applications.MobileNet(input_shape=None,
#                                        alpha=1.0,
#                                        depth_multiplier=1,
#                                        dropout=0.001,
#                                        include_top=True,
#                                        weights="imagenet",
#                                        input_tensor=None,
#                                        pooling=None,
#                                        classes=1000,
#                                        classifier_activation="softmax")

# mnetv2 = tf.keras.applications.MobileNetV2(input_shape=None,
#                                            alpha=1.0,
#                                            include_top=True,
#                                            weights="imagenet",
#                                            input_tensor=None,
#                                            pooling=None,
#                                            classes=1000,
#                                            classifier_activation="softmax")

vgg16 = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

vgg19 = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# print("Starting to read files")
# w1 = np.loadtxt("weights/vgg19_-3.np")
# print("Read file 1")
# w2 = np.loadtxt("weights/vgg16_-3.np")
# print("Read file 2")

# print(
#     analyse_weights(
#         w1, w2,
#         {
#             'fp': [0.01, 0.001], # for different floating point thresholds
#             # 'sim': [.7, .8, .9], # for naive diff similarity percentage
#             'diff': [.1, .2, .3], # for lsh difference
#         },
#         500, 500 # block dims
#         , 512 # bits
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

print(
    analyse_models(
        vgg16,
        vgg19,
        {
            'fp': [0.01, 0.001],  # for different floating point thresholds
            # 'sim': [.7, .8, .9],  # for naive diff similarity percentage
            'diff': [.1, .2, .3], # for lsh difference
        },
        500,
        500,
        32
        , 256
    ))