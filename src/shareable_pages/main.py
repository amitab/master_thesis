from deduplicator import analyse_model_weights
from deduplicator import analyse_models, analyse_models_v2, analyse_models_v2_and_dedup
from deduplicator import analyse_weights

import tensorflow as tf
import numpy as np

# DATA_DRIFT_MODELS_PATH = "../drift/Concept Drift (Data)/"

# m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + 'based_model/based_model-45')
# m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + '30k_normal_added_10k_mix/30k_normal_added_10k_mix-45')

# m1._name = "based_model"
# m2._name = "30k_normal_added_10k_mix"

# mnet = tf.keras.applications.MobileNet()
# mnetv2 = tf.keras.applications.MobileNetV2()
# vgg16 = tf.keras.applications.VGG16()
# vgg19 = tf.keras.applications.VGG19()

# resnet50 = tf.keras.applications.ResNet50()
# resnet101 = tf.keras.applications.ResNet101()
# resnet101v2 = tf.keras.applications.ResNet101V2()
# resnet152 = tf.keras.applications.ResNet152()

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

sizes = range(200, 2000, 100)


for size in sizes:
    vgg16 = tf.keras.applications.VGG16()
    vgg19 = tf.keras.applications.VGG19()

    analyse_models_v2_and_dedup(
        # m1, m2,
        vgg16, vgg19,
        # mnet, mnetv2,
        # resnet50, resnet101,
        {
            'pairwise': {
                'fp': [0.001],  # for different floating point thresholds
                'sim': [.7, .8, .9],  # for naive diff similarity percentage
            },
            # 'cosine': {
            #     'diff': [.1, .2, .3], # for lsh difference
            # }
            # 'l2lsh': {
            #     # 'r': np.arange(0.1, 2, 0.1),
            #     # # 'k': np.arange(1, 30, 1),
            #     # # 'l': np.arange(10, 70, 1),
            #     # 'r': [0.1],
            #     # 'k': [11],
            #     # 'l': [46],

            #     'r': np.arange(0.1, 1, 0.2),
            #     'k': np.arange(1, 20, 4),
            #     'l': np.arange(10, 40, 4),
            #     'd': 1
            # }
        },
        size,
        size,
        8,
        # 'test'
    )