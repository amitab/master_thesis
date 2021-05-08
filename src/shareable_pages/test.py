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
    s1_lsh = [signature_bit(x.flatten(), ref_planes) for x in s1]
    s2_lsh = [signature_bit(x.flatten(), ref_planes) for x in s2]
    print("Complete lsh signature generation")

    info = {t : {} for t in diff_thresholds}
    num_per_block = s1[0].shape[0] * s1[0].shape[1] 

    for i in range(len(s1)):
        sig1 = s1_lsh[i]
        for j in range(i + 1, len(s1)):
            assert s1[i].shape == s1[j].shape
            sig2 = s1_lsh[j]
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if f's1-{i}' not in info[f]:
                    info[f][f's1-{i}'] = []
                if cosine_hash <= f:
                    info[f][f's1-{i}'].append(f's1-{j}')
    print("Complete for s1")

    for i in range(len(s2)):
        sig1 = s2_lsh[i]
        for j in range(i + 1, len(s2)):
            assert s2[i].shape == s2[j].shape
            sig2 = s2_lsh[j]
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if f's2-{i}' not in info[f]:
                    info[f][f's2-{i}'] = []
                if cosine_hash <= f:
                    info[f][f's2-{i}'].append(f's2-{j}')
    print("Complete for s2")

    for i, b1 in enumerate(s1):
        sig1 = s1_lsh[i]
        for j, b2 in enumerate(s2):
            assert b1.shape == b2.shape
            sig2 = s2_lsh[j]
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if f's1-{i}' not in info[f]:
                    info[f][f's1-{i}'] = []
                if cosine_hash <= f:
                    info[f][f's1-{i}'].append(f's2-{j}')
    print("Complete for s1-s2")

    cons = {t : {} for t in diff_thresholds}
    for f in diff_thresholds:
        b_names = list(info[f].keys())

        unique_bs = set()

        for b_name in b_names:
            if not b_name in info[f]:
                continue
            unique_bs.add(b_name)
            pending_bs = set(info[f][b_name])
            pending_bs.add(b_name)
            del info[f][b_name]
            temp_bs = set()
            while pending_bs:
                for similar_b in pending_bs:
                    if not similar_b in info[f]:
                        continue
                    temp_bs.update(info[f][similar_b])
                    del info[f][similar_b]

                pending_bs = temp_bs
                temp_bs = set()

        cons[f] = {
            # 'unique_bs': unique_bs,
            'total_blocks': (len(s1) + len(s2)),
            'num_unique': len(unique_bs),
            'num_reduced': ((len(s1) + len(s2)) - len(unique_bs)),
            'bytes_reduced': ((len(s1) + len(s2)) - len(unique_bs)) * num_per_block * 8,
            'total_bytes': (len(s1) + len(s2)) * num_per_block * 8,
        }

    return cons


def compare_block_sets(s1, s2, sim_thresholds, fp_thresholds):
    info = {
        f: {t : {} for t in sim_thresholds} for f in fp_thresholds
    }

    num_per_block = s1[0].shape[0] * s1[0].shape[1] 

    for i in range(len(s1)):
        for j in range(i + 1, len(s1)):
            assert s1[i].shape == s1[j].shape
            diff = np.absolute(s1[i] - s1[j])
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = s1[i].shape[0] * s1[i].shape[1]
                for t in sim_thresholds:
                    if f's1-{i}' not in info[f][t]:
                        info[f][t][f's1-{i}'] = []
                    if d / tot >= t:
                        info[f][t][f's1-{i}'].append(f's1-{j}')
                        
    print("Complete for s1")

    for i in range(len(s2)):
        for j in range(i + 1, len(s2)):
            assert s2[i].shape == s2[j].shape
            diff = np.absolute(s2[i] - s2[j])
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = s2[i].shape[0] * s2[i].shape[1]
                for t in sim_thresholds:
                    if f's2-{i}' not in info[f][t]:
                        info[f][t][f's2-{i}'] = []
                    if d / tot >= t:
                        info[f][t][f's2-{i}'].append(f's2-{j}')

    print("Complete for s2")

    for i, b1 in enumerate(s1):
        for j, b2 in enumerate(s2):
            assert b1.shape == b2.shape
            diff = np.absolute(b1 - b2)
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = b1.shape[0] * b2.shape[1]
                for t in sim_thresholds:
                    if f's1-{i}' not in info[f][t]:
                        info[f][t][f's1-{i}'] = []
                    if d / tot >= t:
                        info[f][t][f's1-{i}'].append(f's2-{j}')
    print("Complete for s1-s2")

    cons = {
        f: {t : 0 for t in sim_thresholds} for f in fp_thresholds
    }
    for f in fp_thresholds:
        for t in sim_thresholds:
            b_names = list(info[f][t].keys())

            unique_bs = set()

            for b_name in b_names:
                if not b_name in info[f][t]:
                    continue
                unique_bs.add(b_name)
                pending_bs = set(info[f][t][b_name])
                pending_bs.add(b_name)
                del info[f][t][b_name]
                temp_bs = set()
                while pending_bs:
                    for similar_b in pending_bs:
                        if not similar_b in info[f][t]:
                            continue
                        temp_bs.update(info[f][t][similar_b])
                        del info[f][t][similar_b]

                    pending_bs = temp_bs
                    temp_bs = set()

            cons[f][t] = {
                # 'unique_bs': unique_bs,
                'total_blocks': (len(s1) + len(s2)),
                'num_unique': len(unique_bs),
                'num_reduced': ((len(s1) + len(s2)) - len(unique_bs)),
                'bytes_reduced': ((len(s1) + len(s2)) - len(unique_bs)) * num_per_block * 8,
                'total_bytes': (len(s1) + len(s2)) * num_per_block * 8,
            }

    return cons


print("Starting to read files")
w1 = np.loadtxt("vgg19_-3.np")
print("Read file 1")
w2 = np.loadtxt("vgg16_-3.np")
print("Read file 2")

a = min(w1.shape[0], w2.shape[0])
b = min(w1.shape[1], w2.shape[1])

diff = np.absolute(w1[:a,:b] - w2[:a,:b])
d_a = np.count_nonzero(diff <= 0.01)
d_b = np.count_nonzero(diff <= 0.001)

a_p = (d_a / (a * b)) * 100
b_p = (d_b / (a * b)) * 100

print(f"For fp_threshold 0.01, similarity = {a_p}%")
print(f"For fp_threshold 0.001, similarity = {b_p}%")

print(f'W1 shape: {w1.shape[0]} x {w1.shape[1]}')
print(f'W2 shape: {w2.shape[0]} x {w2.shape[1]}')


print("Starting Splitting")
bx = 1000
by = 1000
s1 = split_matrix(w1, bx, by)
print("Split w1")
s2 = split_matrix(w2, bx, by)
print("Split w2")

print("Comparing block sets")
print(compare_block_sets(s1, s2, [.7, .8, .9], [0.01]))
# print(compare_lsh_block_sets(s1, s2, [.1, .2, .3], bx * by, 2046))
print("Done comparing block sets")