import math
import copy

import numpy as np

from tqdm import tqdm

from .blocks import ModelBlocks, WeightBlocks
from .tf_layer_helpers import layer_weight_transformer, layer_bytes


def get_prime_factors(number):
    prime_factors = []

    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2

    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.append(int(i))
            number = number / i
    if number > 2:
        prime_factors.append(int(number))

    return prime_factors


def split_matrix_even(array, nrows, ncols, get_shape=False):
    r, h = array.shape
    if (r <= nrows and h <= ncols) or (r >= nrows and h >= ncols):
        if get_shape:
            ret, shape = split_matrix_v2(array, nrows, ncols)
            return (r, h,), (r, h,), ret, shape
        return (r, h), (r, h,), split_matrix_v2(array, nrows, ncols)

    if h < ncols and r > nrows:
        p, q = r, h
        prim = get_prime_factors(p)

        while p < nrows and q > ncols:
            k = prim.pop()
            q *= k
            p /= k

        p, q = int(p), int(q)
        print(f"Reshaped matrix from {r},{h} -> {p},{q}")
        array = array.reshape(p, q)

        # return array
        if get_shape:
            ret, shape = split_matrix_v2(array, nrows, ncols)
            return (r, h,), (p, q), ret, shape
        return (r, h), (p, q), split_matrix_v2(array, nrows, ncols)


    assert r < nrows and h > ncols

    p, q = r, h
    prim = get_prime_factors(q)

    while p < nrows and q > ncols:
        k = prim.pop()
        p *= k
        q /= k

    p, q = int(p), int(q)
    print(f"Reshaped matrix from {r},{h} -> {p},{q}")
    array = array.reshape(p, q)

    # return array
    if get_shape:
        try:
            ret, shape = split_matrix_v2(array, nrows, ncols)
            return (r, h,), (p, q), ret, shape
        except:
            import pdb
            pdb.set_trace()
    return (r, h), (p, q), split_matrix_v2(array, nrows, ncols)


def split_matrix_v2(array, nrows, ncols):
    assert len(array.shape) == 2

    r, h = array.shape
    assert r != 0 or h != 0

    if r < nrows and h < ncols:
        temp = np.zeros((nrows, ncols))
        temp[:r, :h] = array
        return [(temp, (r,h))], (1, 1,)

    r_mul = math.floor(r / nrows) * nrows
    h_mul = math.floor(h / ncols) * ncols

    r_left = r - r_mul
    assert 0 <= r_left < nrows
    h_left = h - h_mul
    assert 0 <= h_left < ncols

    num_x_blocks = math.ceil(r_mul / float(nrows))
    num_y_blocks = math.ceil(h_mul / float(ncols))

    r_full = (math.ceil(r / nrows) * nrows) if r % nrows != 0 else r
    h_full = (math.ceil(h / ncols) * ncols) if h % ncols != 0 else h

    nx = math.ceil(r_full / float(nrows))
    ny = math.ceil(h_full / float(ncols))

    nrray = array[:r_mul, :h_mul]
    left_row = array[r_mul:, :h_mul]
    left_col = array[:, h_mul:]

    ret = []

    if nrray.size > 0:
        ret.extend(np.vsplit(nrray, num_x_blocks))
        ret = [np.hsplit(r, num_y_blocks) for r in ret]
        ret = [[(block, (nrows, ncols)) for block in row] for row in ret]

    if left_row.size > 0:
        row = []
        height = left_row.shape[0]
        for i in range(num_y_blocks):
            width = min(ncols, left_row.shape[1] - i * ncols)
            temp = np.zeros((nrows, ncols))
            temp[:height, :width] = left_row[:, i * ncols:i * ncols + ncols]
            row.append((temp, (height, width)))
        ret.append(row)

    if left_col.size > 0:
        width = left_col.shape[1]
        for i in range(nx):
            height = min(nrows, left_col.shape[0] - i * nrows)
            temp = np.zeros((nrows, ncols))
            temp[:height, :width] = left_col[i * nrows:i * nrows + nrows, :]
            if i >= len(ret):
                ret.append([(temp, (height, width))])
            else:
                ret[i].append((temp, (height, width)))

    ret = [j for i in ret for j in i]

    if len(ret) != nx * ny:
        import pdb
        pdb.set_trace()

    assert len(ret) == nx * ny
    assert isinstance(ret[0], tuple) and isinstance(ret[0][0], np.ndarray)

    return ret, (
        nx,
        ny,
    )


def split_matrix(array, nrows, ncols, get_shape=False):
    assert len(array.shape) == 2

    r, h = array.shape
    assert r != 0 or h != 0

    if r < nrows and h < ncols:
        temp = np.zeros((nrows, ncols))
        temp[:r, :h] = array
        if get_shape:
            return [temp], (1, 1,)
        return [temp]

    r_mul = math.floor(r / nrows) * nrows
    h_mul = math.floor(h / ncols) * ncols

    r_left = r - r_mul
    assert 0 <= r_left < nrows
    h_left = h - h_mul
    assert 0 <= h_left < ncols

    num_x_blocks = math.ceil(r_mul / float(nrows))
    num_y_blocks = math.ceil(h_mul / float(ncols))

    r_full = (math.ceil(r / nrows) * nrows) if r % nrows != 0 else r
    h_full = (math.ceil(h / ncols) * ncols) if h % ncols != 0 else h

    nx = math.ceil(r_full / float(nrows))
    ny = math.ceil(h_full / float(ncols))

    nrray = array[:r_mul, :h_mul]
    left_row = array[r_mul:, :h_mul]
    left_col = array[:, h_mul:]

    ret = []

    if nrray.size > 0:
        ret.extend(np.vsplit(nrray, num_x_blocks))
        ret = [np.hsplit(r, num_y_blocks) for r in ret]

    if left_row.size > 0:
        row = []
        height = left_row.shape[0]
        for i in range(num_y_blocks):
            width = min(ncols, left_row.shape[1] - i * ncols)
            temp = np.zeros((nrows, ncols))
            temp[:height, :width] = left_row[:, i * ncols:i * ncols + ncols]
            row.append(temp)
        ret.append(row)

    if left_col.size > 0:
        width = left_col.shape[1]
        for i in range(nx):
            height = min(nrows, left_col.shape[0] - i * nrows)
            temp = np.zeros((nrows, ncols))
            temp[:height, :width] = left_col[i * nrows:i * nrows + nrows, :]
            if i >= len(ret):
                ret.append([temp])
            else:
                ret[i].append(temp)

    ret = [j for i in ret for j in i]

    if len(ret) != nx * ny:
        import pdb
        pdb.set_trace()

    assert len(ret) == nx * ny
    assert isinstance(ret[0], np.ndarray)

    if get_shape:
        return ret, (
            nx,
            ny,
        )
    return ret


def split(array, nrows, ncols):
    if len(array.shape) == 1:
        return split_vector(array, nrows, ncols)
    else:
        return split_matrix(array, nrows, ncols)


def split_vector(array, nrows, ncols, get_shape=False):
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
    if get_shape:
        return [x.reshape(nrows, ncols) for x in ret], (
            1,
            num_blocks,
        )
    return [x.reshape(nrows, ncols) for x in ret]


def split_weight(array, nrows, ncols, name, w_idx):
    if len(array.shape) == 1:
        array = array.reshape(-1,1)

    r, h = array.shape
    # if len(array.shape) == 1:
    #     ret, shape = split_vector(array, nrows, ncols, True)
    #     wblocks = WeightBlocks(shape[0], shape[1], name, w_idx, (r, h),
    #                            True)
    #     for i in range(shape[1]):
    #         wblocks.add_block(0, i, ret[i])
    #     return wblocks
    # else:
    old_shape, new_shape, ret, blocking_shape = split_matrix_even(array, nrows, ncols, True)

    wblocks = WeightBlocks(blocking_shape, name, w_idx, new_shape, False, old_shape)
    for i in range(blocking_shape[0]):
        for j in range(blocking_shape[1]):
            idx = i * blocking_shape[1] + j
            wblocks.add_block(i, j, ret[idx][0], ret[idx][1])
    return wblocks


def split_model(m, nrows, ncols, weight_lower_bound=32):
    b = ModelBlocks()
    for idx in tqdm(range(len(m.layers)), desc="Splitting model into blocks"):
        l = m.layers[idx]
        if len(l.weights) == 0 or layer_bytes(l) < weight_lower_bound:
            continue

        b.add_weight(
            split_weight(layer_weight_transformer(l), nrows, ncols, l.name,
                         idx), idx)

    # b.finalize()
    return b


def dedup_blocks(mapping, mb1, mb2):
    omb1 = copy.deepcopy(mb1)
    omb2 = copy.deepcopy(mb2)

    for m in mapping:
        if mapping[m] is not None and m.startswith('s1'):
            a_set, idx_a = m.split('-')
            idx_a = int(idx_a)
            b_set, idx_b = mapping[m].split('-')
            idx_b = int(idx_b)

            omb1[idx_a] = omb1[idx_b] if b_set == a_set else omb2[idx_b]

    for m in mapping:
        if mapping[m] is not None and m.startswith('s2'):
            a_set, idx_a = m.split('-')
            idx_a = int(idx_a)
            b_set, idx_b = mapping[m].split('-')
            idx_b = int(idx_b)

            omb2[idx_a] = omb2[idx_b] if b_set == a_set else omb1[idx_b]

    return omb1, omb2


def verify_dedup_blocks(mapping, mb1, mb2, omb1, omb2):
    for m in mapping:
        if mapping[m] is not None and m.startswith('s1'):
            a_set, idx_a = m.split('-')
            idx_a = int(idx_a)
            b_set, idx_b = mapping[m].split('-')
            idx_b = int(idx_b)

            if b_set == a_set:
                # The blocks at the edges of a matrix could have different shapes
                aups = mb1.getBlock(idx_a).unpadded_shape
                bups = mb1.getBlock(idx_b).unpadded_shape
                ax = min(aups[0], bups[0])
                bx = min(aups[1], bups[1])
                assert np.array_equal(mb1[idx_a][:ax,:bx], mb1[idx_b][:ax,:bx])
            else:
                aups = mb1.getBlock(idx_a).unpadded_shape
                bups = mb2.getBlock(idx_b).unpadded_shape
                ax = min(aups[0], bups[0])
                bx = min(aups[1], bups[1])
                assert np.array_equal(mb1[idx_a][:ax,:bx], mb2[idx_b][:ax,:bx])

            assert not np.array_equal(mb1[idx_a], omb1[idx_a])

    for m in mapping:
        if mapping[m] is not None and m.startswith('s2'):
            a_set, idx_a = m.split('-')
            idx_a = int(idx_a)
            b_set, idx_b = mapping[m].split('-')
            idx_b = int(idx_b)

            if b_set == a_set:
                aups = mb2.getBlock(idx_a).unpadded_shape
                bups = mb2.getBlock(idx_b).unpadded_shape
                ax = min(aups[0], bups[0])
                bx = min(aups[1], bups[1])
                assert np.array_equal(mb2[idx_a][:ax,:bx], mb2[idx_b][:ax,:bx])
            else:
                aups = mb2.getBlock(idx_a).unpadded_shape
                bups = mb1.getBlock(idx_b).unpadded_shape
                ax = min(aups[0], bups[0])
                bx = min(aups[1], bups[1])
                assert np.array_equal(mb2[idx_a][:ax,:bx], mb1[idx_b][:ax,:bx])
            
            assert not np.array_equal(mb2[idx_a], omb2[idx_a])
