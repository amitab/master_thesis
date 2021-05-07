from __future__ import division

import tensorflow as tf
import numpy as np

import math

def signature_bit(data, planes):
    """
	LSH signature generation using random projection
	Returns the signature bits for two data points.
	The signature bits of the two points are different
 	only for the plane that divides the two points.
 	"""
    sig = 0
    for p in planes:
        sig <<= 1
        if np.dot(data, p) >= 0:
            sig |= 1
    return sig


def bitcount(n):
    """
	gets the number of bits set to 1
	"""
    count = 0
    while n:
        count += 1
        n = n & (n - 1)
    return count


def length(v):
    """returns the length of a vector"""
    return math.sqrt(np.dot(v, v))


def split_matrix(array, nrows, ncols):
    assert len(array.shape) == 2
    """Split a matrix into sub-matrices."""
    r, h = array.shape
    if r % nrows != 0:
        padding = (math.ceil(r / nrows) * nrows) - r
        array = np.vstack((array, np.zeros((padding, h))))
        r, h = array.shape
    if h % ncols != 0:
        padding = (math.ceil(h / ncols) * ncols) - h
        array = np.hstack((array, np.zeros((r, padding))))
        r, h = array.shape
    num_x_blocks = math.ceil(r / float(nrows))
    num_y_blocks = math.ceil(h / float(ncols))

    rows = np.vsplit(array, num_x_blocks)
    blocks = [np.hsplit(row, num_y_blocks) for row in rows]
    ret = [j for i in blocks for j in i]

    assert len(ret) == num_x_blocks * num_y_blocks
    assert isinstance(ret[0], np.ndarray)
    return ret


def compare_lsh_block_sets(s1, s2, diff_thresholds, dim, bits):
    ref_planes = np.random.randn(bits, dim)

    info = {
        's1': {k: 0
               for k in diff_thresholds},
        's2': {k: 0
               for k in diff_thresholds},
        's1-s2': {k: 0
                  for k in diff_thresholds},
    }

    for i in range(len(s1)):
        sig1 = signature_bit(s1[i].flatten(), ref_planes)
        for j in range(i + 1, len(s1)):
            assert s1[i].shape == s1[j].shape
            sig2 = signature_bit(s1[j].flatten(), ref_planes)
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if cosine_hash <= f:
                    info['s1'][f] += 1
    print("Complete for s1")

    for i in range(len(s2)):
        sig1 = signature_bit(s2[i].flatten(), ref_planes)
        for j in range(i + 1, len(s2)):
            assert s2[i].shape == s2[j].shape
            sig2 = signature_bit(s2[j].flatten(), ref_planes)
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if cosine_hash <= f:
                    info['s2'][f] += 1
    print("Complete for s2")

    for b1 in s1:
        sig1 = signature_bit(b1.flatten(), ref_planes)
        for b2 in s2:
            assert b1.shape == b2.shape
            sig2 = signature_bit(b2.flatten(), ref_planes)
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if cosine_hash <= f:
                    info['s1-s2'][f] += 1
    print("Complete for s1-s2")

    return info


def compare_block_sets(s1, s2, sim_thresholds, fp_thresholds):
    info = {
        's1': {f: {k: 0
                   for k in sim_thresholds}
               for f in fp_thresholds},
        's2': {f: {k: 0
                   for k in sim_thresholds}
               for f in fp_thresholds},
        's1-s2': {f: {k: 0
                      for k in sim_thresholds}
                  for f in fp_thresholds}
    }

    s1_sets = {}
    for i in range(len(s1)):
        s1_sets[i] = {f: {k: []
                      for k in sim_thresholds}
                  for f in fp_thresholds}
        for j in range(i + 1, len(s1)):
            assert s1[i].shape == s1[j].shape
            diff = np.absolute(s1[i] - s1[j])
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = s1[i].shape[0] * s1[i].shape[1]
                for t in sim_thresholds:
                    if d / tot >= t:
                        info['s1'][f][t] += 1
                        s1_sets[i][f][t].append(j)
    print("Complete for s1")
    import pdb
    pdb.set_trace()

    s2_sets = {}
    for i in range(len(s2)):
        s2_sets[i] = {f: {k: []
                      for k in sim_thresholds}
                  for f in fp_thresholds}
        for j in range(i + 1, len(s2)):
            assert s2[i].shape == s2[j].shape
            diff = np.absolute(s2[i] - s2[j])
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = s2[i].shape[0] * s2[i].shape[1]
                for t in sim_thresholds:
                    if d / tot >= t:
                        info['s2'][f][t] += 1
                        s2_sets[i][f][t].append(j)
    print("Complete for s2")
    import pdb
    pdb.set_trace()

    for b1 in s1:
        for b2 in s2:
            assert b1.shape == b2.shape
            diff = np.absolute(b1 - b2)
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = b1.shape[0] * b2.shape[1]
                for t in sim_thresholds:
                    if d / tot >= t:
                        info['s1-s2'][f][t] += 1
    print("Complete for s1-s2")

    return info


print("Starting to read files")
w1 = np.loadtxt("vgg19_-1.np")
print("Read file 1")
w2 = np.loadtxt("resnet152_-1.np")
print("Read file 2")

print("Starting Splitting")
bx = 100
by = 100
s1 = split_matrix(w1, bx, by)
print("Split w1")
s2 = split_matrix(w2, bx, by)
print("Split w2")

print("Comparing block sets")
print(compare_block_sets(s1, s2, [.7, .8, .9], [0.01]))
# print(compare_lsh_block_sets(s1, s2, [.1, .2, .3, .4, .5, .6, .7, .8, .9], bx * by, 1024))
print("Done comparing block sets")