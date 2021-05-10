from datasketch import MinHash, MinHashLSH
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from matrix_utils import split, split_model, dedup_blocks

vgg16 = tf.keras.applications.VGG16()
vgg19 = tf.keras.applications.VGG19()

s1 = split_model(vgg16, 500, 500, 32)
s2 = split_model(vgg19, 500, 500, 32)

num_perm = 128
min_dict = {}

pbar = tqdm(total=len(s1))
for i, val in enumerate(s1):
   m = MinHash(num_perm=num_perm)
   m.update_batch(np.floor(val.flatten() * 10000).astype(int))
   min_dict[f"s1-{i}"] = m
   pbar.update(1)
#    if i > 20:
#        break

pbar = tqdm(total=len(s2))
for i, val in enumerate(s2):
   m = MinHash(num_perm=num_perm)
   m.update_batch(np.floor(val.flatten() * 10000).astype(int))
   min_dict[f"s2-{i}"] = m
   pbar.update(1)
#    if i > 20:
#        break


lsh2 = MinHashLSH(threshold=0.9, num_perm=num_perm)
for key in min_dict.keys():
   lsh2.insert(key, min_dict[key])

test = {k: lsh2.query(min_dict[k]) for k in min_dict if len(lsh2.query(min_dict[k])) > 1}

import pdb
pdb.set_trace()