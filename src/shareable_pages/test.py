from __future__ import division

import tensorflow as tf
import numpy as np

from tqdm import tqdm

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


# Memory Fusion from Multiple Neural Networks
def conv2d_transformer(w):
    dims = w.shape
    if not len(dims) == 4:
        import pdb
        pdb.set_trace()
    assert len(dims) == 4

    return w.reshape(dims[0], dims[1] * dims[2] * dims[3])


layer_weights_transformer_mapping = {'conv2d': conv2d_transformer}


def layer_weight_transformer(layer):
    if len(layer.weights[0].shape) == 4:
        return layer_weights_transformer_mapping['conv2d'](
            layer.weights[0].numpy())
    if len(layer.weights[0].shape) > 2:
        import pdb
        pdb.set_trace()
    assert len(layer.weights[0].shape) <= 2
    return layer.weights[0].numpy()


def split(array, nrows, ncols):
    if len(array.shape) == 1:
        return split_vector(array, nrows, ncols)
    else:
        return split_matrix(array, nrows, ncols)


def split_vector(array, nrows, ncols):
    assert len(array.shape) == 1
    l = array.shape[0]
    block_size = nrows * ncols
    num_blocks = math.ceil(l / float(block_size))
    if l % block_size != 0:
        new_arr = np.zeros(num_blocks * block_size)
        new_arr[:l] = array[:l]
        array = new_arr
    ret = np.hsplit(array, num_blocks)

    assert len(ret) == num_blocks
    assert isinstance(ret[0], np.ndarray)
    # Returning 2 d array
    return [x.reshape(nrows, ncols) for x in ret]


def compare_lsh_block_sets(s1, s2, diff_thresholds, dim, bits):
    ref_planes = np.random.randn(bits, dim)
    s1_lsh = [signature_bit(x.flatten(), ref_planes) for x in tqdm(s1)]
    s2_lsh = [signature_bit(x.flatten(), ref_planes) for x in tqdm(s2)]
    print("Complete lsh signature generation")

    info = {t: {} for t in diff_thresholds}
    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    for i in tqdm(range(len(s1))):
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

    for i in tqdm(range(len(s2))):
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

    for i, b1 in tqdm(enumerate(s1)):
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

    cons = {t: {} for t in diff_thresholds}
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
            'num_unique':
            len(unique_bs),
            'num_reduced': ((len(s1) + len(s2)) - len(unique_bs)),
            'bytes_reduced':
            ((len(s1) + len(s2)) - len(unique_bs)) * num_per_block * 8,
            'total_bytes': (len(s1) + len(s2)) * num_per_block * 8,
        }

    return cons


def compare_block_sets(s1, s2, sim_thresholds, fp_thresholds):
    info = {f: {t: {} for t in sim_thresholds} for f in fp_thresholds}

    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    for i in tqdm(range(len(s1))):
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

    for i in tqdm(range(len(s2))):
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

    for i, b1 in tqdm(enumerate(s1)):
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

    print("Generating unique block counts...")
    cons = {f: {t: 0 for t in sim_thresholds} for f in fp_thresholds}
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
                'num_unique':
                len(unique_bs),
                'num_reduced': ((len(s1) + len(s2)) - len(unique_bs)),
                'bytes_reduced':
                ((len(s1) + len(s2)) - len(unique_bs)) * num_per_block * 8,
                'total_bytes': (len(s1) + len(s2)) * num_per_block * 8,
            }

    return cons


def analyse_weights(w1, w2, thresholds, bx, by, bits=None):
    assert len(w1.shape) == len(w2.shape) == 2

    print(f'W1 shape: {w1.shape[0]} x {w1.shape[1]}')
    print(f'W2 shape: {w2.shape[0]} x {w2.shape[1]}')

    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    a = min(w1.shape[0], w2.shape[0])
    b = min(w1.shape[1], w2.shape[1])

    diff = np.absolute(w1[:a, :b] - w2[:a, :b])
    for f in fp_thresholds:
        d = np.count_nonzero(diff <= f)
        p = (d / (a * b)) * 100
        print(f"For fp_threshold {f}, similarity = {p}%")

    print("Starting Splitting")
    s1 = split_matrix(w1, bx, by)
    print("Split w1")
    s2 = split_matrix(w2, bx, by)
    print("Split w2")

    if 'sim' in thresholds:
        # Compare blocks, naive
        sim_thresholds = thresholds['sim']
        print("Comparing block sets")
        return compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)
    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        assert dims != None
        assert bits != None
        print("Comparing LSH")
        return compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)


def analyse_model_weights(m1, m2, thresholds):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    weights1 = [(l.name, layer_weight_transformer(l)) for l in m1.layers
                if len(l.get_weights()) != 0]
    weights2 = [(l.name, layer_weight_transformer(l)) for l in m2.layers
                if len(l.get_weights()) != 0]

    w_info = {}
    s1 = []
    s2 = []

    for i, x in enumerate(weights1):
        w1_name, w1 = x
        w_info[f'w1{i}'] = {'shape': w1.shape, 'name': w1_name, 'vs': {}}
        for j, y in enumerate(weights2):
            w2_name, w2 = y
            print(f"Comparing m1{i} ({w1_name}) with m2{j} ({w2_name})...")
            w_info[f'w1{i}']['vs'][f'w2{j}'] = {}
            data = w_info[f'w1{i}']['vs'][f'w2{j}']
            data['shape'] = w2.shape
            data['name'] = w2_name
            data['match'] = {}

            smol = None
            big = None
            k = None
            l = None
            if w1.flatten().shape[0] < w2.flatten().shape[0]:
                smol = w1.flatten()
                big = w2.flatten()
                k = smol.shape[0]
                l = big.shape[0]
            else:
                smol = w2.flatten()
                big = w1.flatten()
                k = smol.shape[0]
                l = big.shape[0]

            for f in fp_thresholds:
                best_match = 0
                for m in range(k, l):
                    diff = np.absolute(smol - big[m - k:m])
                    d = np.count_nonzero(diff <= f)
                    best_match = max(d, best_match)
                p = (best_match / k) * 100
                data['match'][f] = p

    return w_info


def analyse_models(m1, m2, thresholds, bx, by, bits=None):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    # all matrices get transformed to <= 2 dimensions
    # right now only conv2d layers can change to 2d
    weights1 = [
        layer_weight_transformer(l) for l in m1.layers if len(l.weights) != 0
    ]
    weights2 = [
        layer_weight_transformer(l) for l in m2.layers if len(l.weights) != 0
    ]

    s1 = []
    s2 = []

    # Even 1d vectors change to 2d blocks
    # Because when calculating similarity between blocks, we can reshape for improving chances of similarity
    print("Starting Splitting")
    for w1 in weights1:
        s1.extend(split(w1, bx, by))
    print("Split m1")
    for w2 in weights2:
        s2.extend(split(w2, bx, by))
    print("Split m2")

    if 'sim' in thresholds:
        # Compare blocks, naive
        sim_thresholds = thresholds['sim']
        print("Comparing block sets")
        return compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)
    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        dims = bx * by
        assert bits != None
        print("Comparing LSH")
        return compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)


mnet = tf.keras.applications.MobileNet(input_shape=None,
                                       alpha=1.0,
                                       depth_multiplier=1,
                                       dropout=0.001,
                                       include_top=True,
                                       weights="imagenet",
                                       input_tensor=None,
                                       pooling=None,
                                       classes=1000,
                                       classifier_activation="softmax")

mnetv2 = tf.keras.applications.MobileNetV2(input_shape=None,
                                           alpha=1.0,
                                           include_top=True,
                                           weights="imagenet",
                                           input_tensor=None,
                                           pooling=None,
                                           classes=1000,
                                           classifier_activation="softmax")

# print("Starting to read files")
# w1 = np.loadtxt("vgg19_-3.np")
# print("Read file 1")
# w2 = np.loadtxt("vgg16_-3.np")
# print("Read file 2")

# print(
#     analyse_weights(
#         w1, w2,
#         {
#             'fp': [0.01, 0.001], # for different floating point thresholds
#             'sim': [.7, .8, .9], # for naive diff similarity percentage
#             # 'diff': [.1, .2, .3], # for lsh difference
#         },
#         500, 500 # block dims
#         # , 2046 # bits
#     )
# )

# print(
#     analyse_model_weights(
#         mnet,
#         mnetv2,
#         {
#             'fp': [0.01, 0.001],  # for different floating point thresholds
#         }
#     ))

print(
    analyse_models(
        mnet,
        mnetv2,
        {
            'fp': [0.01, 0.001],  # for different floating point thresholds
            # 'sim': [.7, .8, .9],  # for naive diff similarity percentage
            'diff': [.1, .2, .3], # for lsh difference
        },
        50,
        50
        , 512
    ))