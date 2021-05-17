import tensorflow as tf
from tf_layer_helpers import layer_weight_transformer, layer_bytes
import numpy as np

import sys
import pickle

from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm

m1 = tf.keras.applications.VGG19()
m2 = tf.keras.applications.VGG16()

# DATA_DRIFT_MODELS_PATH = "../drift/Concept Drift (Data)/"

# m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + 'based_model/based_model-45')
# m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + '30k_normal_added_10k_mix/30k_normal_added_10k_mix-45')

# m1._name = "based_model"
# m2._name = "30k_normal_added_10k_mix"

cache_files = [f for f in listdir("./cache") if isfile(join("./cache/", f)) and f.endswith(".p")]
pbar = tqdm(total=len(cache_files))
for f in cache_files:
    info = pickle.load(open(join("./cache/", f), "rb"))
    d = f.rsplit("_", 5)
    weight_lower_bound = float(d[3])

    new_keys = [m1.name, m2.name, 'all', 'data']
    if new_keys == list(info.keys()):
        continue

    stats = {
        m1.name: {
            'total_size': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m1.layers if len(l.weights) > 0]]) * 8,
            'size_considered': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m1.layers if layer_bytes(l) >= weight_lower_bound]]) * 8,
        },
        m2.name: {
            'total_size': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m2.layers if len(l.weights) > 0]]) * 8,
            'size_considered': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m2.layers if layer_bytes(l) >= weight_lower_bound]]) * 8,
        },
        'all': {}
    }

    if 'data' in info:
        stats.update(info)
    else:
        stats['data'] = info

    stats['all']['total_size'] = stats[m1.name]['total_size'] + stats[m1.name]['total_size']
    stats['all']['size_considered'] = stats[m1.name]['size_considered'] + stats[m1.name]['size_considered']

    pickle.dump(stats, open(join("./cache/", f), "wb"))
    pbar.update(1)