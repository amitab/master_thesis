import tensorflow as tf
import numpy as np

from analyse_model_weights import analyse_model_weights
from analyse_models import analyse_models, analyse_models_v2, analyse_models_v2_and_dedup
from analyse_weights import analyse_weights

DATA_DRIFT_MODELS_PATH = "../drift/Concept Drift (Data)/"

m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + 'based_model/based_model-45')
m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + '30k_normal_added_10k_mix/30k_normal_added_10k_mix-45')

# m1._name = "based_model"
# m2._name = "30k_normal_added_10k_mix"

# mnet = tf.keras.applications.MobileNet()
# mnetv2 = tf.keras.applications.MobileNetV2()
# import pdb
# pdb.set_trace()
# vgg16 = tf.keras.applications.VGG16()
# vgg19 = tf.keras.applications.VGG19()

# resnet50 = tf.keras.applications.ResNet50()
# resnet101 = tf.keras.applications.ResNet101()
# resnet101v2 = tf.keras.applications.ResNet101V2()
# resnet152 = tf.keras.applications.ResNet152()

# import pdb
# pdb.set_trace()

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

print(
    analyse_models_v2_and_dedup(
        m1, m2,
        # vgg16, vgg19,
        # mnet, mnetv2,
        # resnet50, resnet101,
        {
            # 'pairwise': {
            #     'fp': [0.01, 0.001],  # for different floating point thresholds
            #     'sim': [.7, .8, .9],  # for naive diff similarity percentage
            # },
            # 'cosine': {
            #     'diff': [.1, .2, .3], # for lsh difference
            # }
            'l2lsh': {
                # 'r': np.arange(0.1, 2, 0.1),
                # # 'k': np.arange(1, 30, 1),
                # # 'l': np.arange(10, 70, 1),
                # 'k': [11],
                # 'l': [46],

                'r': np.arange(0.1, 2, 0.1),
                'k': np.arange(1, 30, 2),
                'l': np.arange(10, 70, 2),

                'fps': [0.01, 0.001],  # for different floating point thresholds
                'sims': [.7, .8, .9],  # for naive diff similarity percentage
            }
        },
        25,
        25,
        0.1
    ))