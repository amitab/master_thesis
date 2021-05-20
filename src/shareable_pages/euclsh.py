from __future__ import division
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from matrix_utils import split, split_model, dedup_blocks
import math
from collections import defaultdict
from itertools import product

import random
import l2lsh
 
def l2lsh_with_data(bins, r, num_k, num_l, data, seed=None):
    """Create a L2LSH object with given parameters, then insert the data 
    into the object.
    
    Args:
        bins (int): number of bins
        r (float): r is a real positive number
        num_l (int, optional): number of band,
            default=40
        num_k (int, optional): number of concatenated hash function, 
            default=4
        data (np.array): data input
    """
    lsh = l2lsh.L2LSH(bins, r=r, num_k=num_k, num_l=num_l, seed=seed)
    # Insert all data into lsh hash table
    for key in data.keys():
        prob = utils.data_to_probability(data[key], bins)
        sqrt_prob = np.sqrt(prob).reshape(-1, 1)      
        lsh.insert(sqrt_prob, key)
        
    return lsh

vgg16 = tf.keras.applications.VGG16()
vgg19 = tf.keras.applications.VGG19()

s1 = split_model(vgg16, 1300, 1300, 32)
s2 = split_model(vgg19, 1300, 1300, 32)

# Hyper param search


r_range = np.arange(0.001, 1, 0.01)
k_range = np.arange(1, 30, 1)
l_range = np.arange(10, 70, 1)
bins = 1300 * 1300

parameter_combination = list(product(r_range, k_range, l_range))

for params in tqdm(parameter_combination):
    r = params[0]
    k = params[1]
    l = params[2]

    lsh = l2lsh.L2LSH(bins, r=r, num_k=k, num_l=l, seed=10)

    for i, val in enumerate(s1):
        lsh.insert(val.flatten(), f"s1-{i}")

    for i, val in enumerate(s2):
        lsh.insert(val.flatten(), f"s2-{i}")

    if not all([len(v) == 1 for v in lsh.hash_table.values()]):
        test = {k:v for k,v in lsh.hash_table.items() if len(v) > 1}
        import pdb
        pdb.set_trace()