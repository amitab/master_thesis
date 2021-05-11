import math

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from tf_layer_helpers import layer_weight_transformer, layer_bytes

import copy


def idx_to_2d(num_y_blocks, idx):
    i = math.floor(idx / num_y_blocks)
    j = idx % num_y_blocks
    return (
        i,
        j,
    )


class Block(object):
    def __init__(self, block, i, j, num_x_blocks, num_y_blocks, name, w_idx,
                 is_vec):
        self.block = block
        self.i = i
        self.j = j
        self.num_x_blocks = num_x_blocks
        self.num_y_blocks = num_y_blocks
        self.name = name
        self.w_idx = w_idx
        self.is_vec = is_vec


class WeightBlocks(object):
    def __init__(self, num_x_blocks, num_y_blocks, name, w_idx, og_shape,
                 is_vec, reshape=None):
        self.num_x_blocks = num_x_blocks
        self.num_y_blocks = num_y_blocks
        self.name = name
        self.w_idx = w_idx
        self.blocks = {}
        # self.all_blocks = []
        self.is_vec = is_vec
        self.set_size = num_x_blocks * num_y_blocks
        self.og_shape = og_shape
        self.reshape = reshape

        self.cur_i = 0
        self.cur_j = 0

    def add_block(self, i, j, block):
        assert i not in self.blocks or j not in self.blocks[i]
        if i not in self.blocks:
            self.blocks[i] = {}

        self.blocks[i][j] = Block(block, i, j, self.num_x_blocks,
                                  self.num_y_blocks, self.name, self.w_idx,
                                  self.is_vec)
        # self.all_blocks.append(block)

    def __getitem__(self, idx):
        i, j = idx_to_2d(self.num_y_blocks, idx)
        return self.blocks[i][j].block

    def __setitem__(self, idx, item):
        i, j = idx_to_2d(self.num_y_blocks, idx)
        self.blocks[i][j].block = item

    def __iter__(self):
        self.cur_i = 0
        self.cur_j = 0
        return self

    def __next__(self):
        if self.cur_i * self.num_y_blocks + self.cur_j >= self.num_x_blocks * self.num_y_blocks:
            raise StopIteration

        ret = self.blocks[self.cur_i][self.cur_j].block
        if self.cur_j + 1 >= self.num_y_blocks:
            self.cur_i += 1
            self.cur_j = 0
        else:
            self.cur_j += 1
        return ret

    def numpy(self):
        rows = (np.hstack(
            list(self.blocks[i][j].block for j in range(self.num_y_blocks)))
                for i in range(self.num_x_blocks))
        matrix = np.vstack(list(rows))
        matrix = matrix[:self.og_shape[0], :self.og_shape[1]]

        if self.reshape is not None:
            # import pdb
            # pdb.set_trace()
            matrix.reshape(self.reshape)
        
        return matrix


class ModelBlocks(object):
    def __init__(self):
        self.weight_blocks = {}
        self.idxs = []
        self.block_ptrs = {}
        # self.all_blocks = []

        self.cur_w_idx = 0
        self.cur_w = None

        self.set_size = 0

    # def finalize(self):
    #     self.all_blocks = np.array(self.all_blocks)

    def add_weight(self, weight, idx):
        self.weight_blocks[idx] = weight
        # self.all_blocks.extend(weight.all_blocks)
        self.idxs.append(idx)

        self.block_ptrs.update({
            k: (self.set_size, idx)
            for k in range(self.set_size, self.set_size + weight.set_size)
        })
        self.set_size += weight.set_size

    def __len__(self):
        return self.set_size

    def __getitem__(self, idx):
        start, w_idx = self.block_ptrs[idx]
        b_idx = idx - start
        return self.weight_blocks[w_idx][b_idx]

    def __setitem__(self, idx, item):
        start, w_idx = self.block_ptrs[idx]
        b_idx = idx - start
        self.weight_blocks[w_idx][b_idx] = item

    def __iter__(self):
        self.cur_w_idx = 0
        if self.cur_w_idx < len(self.idxs):
            self.cur_w = iter(self.weight_blocks[self.idxs[self.cur_w_idx]])

        return self

    def __next__(self):
        while self.cur_w_idx < len(self.idxs):
            try:
                ret = next(self.cur_w)
                return ret
            except StopIteration:
                self.cur_w_idx += 1
                if self.cur_w_idx < len(self.idxs):
                    self.cur_w = iter(
                        self.weight_blocks[self.idxs[self.cur_w_idx]])
                pass
        raise StopIteration

    def reconstruct(self):
        for idx in self.idxs:
            yield idx, self.weight_blocks[idx].numpy()

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
    if (r < nrows and h < ncols) or (r > nrows and h > ncols):
        if get_shape:
            ret, shape = split_matrix(array, nrows, ncols, get_shape)
            return (r, h,), (r, h,), ret, shape
        return (r, h), (r, h,), split_matrix(array, nrows, ncols, get_shape)

    if h < ncols and r > nrows:
        return split_matrix_even(array, nrows, ncols, get_shape)
    
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
        ret, shape = split_matrix(array, nrows, ncols, get_shape)
        return (r, h,), (p, q), ret, shape
    return (r, h), (p, q), split_matrix(array, nrows, ncols, get_shape)


def split_matrix(array, nrows, ncols, get_shape=False):
    assert len(array.shape) == 2

    r, h = array.shape
    assert r != 0 or h != 0

    if r < nrows and h < ncols:
        temp = np.zeros((nrows, ncols))
        temp[:r, :h] = array
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
    r, h = array.shape
    if len(array.shape) == 1:
        ret, shape = split_vector(array, nrows, ncols, True)
        wblocks = WeightBlocks(shape[0], shape[1], name, w_idx, (r, h),
                               True)
        for i in range(shape[1]):
            wblocks.add_block(0, i, ret[i])
        return wblocks
    else:
        old_shape, new_shape, ret, shape = split_matrix_even(array, nrows, ncols, True)
        wblocks = WeightBlocks(shape[0], shape[1], name, w_idx, new_shape,
                               False, old_shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                idx = i * shape[1] + j
                wblocks.add_block(i, j, ret[idx])
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


def test_iteration():
    print("Running iteration test")
    m = tf.keras.applications.VGG16()
    weight_lower_bound = 32
    bx = 500
    by = 500

    weights = [
        layer_weight_transformer(l) for l in m.layers
        if len(l.weights) != 0 and layer_bytes(l) >= weight_lower_bound
    ]

    assert len(weights) > 0

    sa = []
    for w in tqdm(weights, desc="Splitting model into blocks"):
        sa.extend(split(w, bx, by))

    sb = split_model(m, bx, by, weight_lower_bound)

    assert len(sa) == len(sb)
    for i in range(len(sa)):
        assert np.array_equal(sa[i], sb[i])


def test_split():
    print("Running model split test")
    m = tf.keras.applications.VGG16()
    weight_lower_bound = 16
    bx = 500
    by = 500

    sb = split_model(m, bx, by, weight_lower_bound)

    re = sb.reconstruct()
    for k in tqdm(re, desc="Comparing reconstructed blocks"):
        w = layer_weight_transformer(m.layers[k])
        if not np.array_equal(w, re[k]):
            import pdb
            pdb.set_trace()
        assert np.array_equal(w, re[k])


# Test case
if __name__ == "__main__":
    test_iteration()
    test_split()