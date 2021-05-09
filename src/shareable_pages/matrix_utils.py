import math

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from tf_layer_helpers import layer_weight_transformer, layer_bytes

import copy

def idx_to_2d(num_y_blocks, idx):
    i = math.floor(idx / num_y_blocks)
    j = idx % num_y_blocks
    return (i, j,)

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
    def __init__(self, num_x_blocks, num_y_blocks, name, w_idx, og_shape, is_vec):
        self.num_x_blocks = num_x_blocks
        self.num_y_blocks = num_y_blocks
        self.name = name
        self.w_idx = w_idx
        self.blocks = {}
        self.is_vec = is_vec
        self.set_size = num_x_blocks * num_y_blocks
        self.og_shape = og_shape

        self.cur_i = 0
        self.cur_j = 0

    def add_block(self, i, j, block):
        assert i not in self.blocks or j not in self.blocks[i]
        if i not in self.blocks:
            self.blocks[i] = {}

        self.blocks[i][j] = Block(block, i, j, self.num_x_blocks,
                                  self.num_y_blocks, self.name, self.w_idx,
                                  self.is_vec)

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
        rows = (np.hstack(list(self.blocks[i][j].block for j in range(self.num_y_blocks))) for i in range(self.num_x_blocks))
        matrix = np.vstack(list(rows))
        return matrix[:self.og_shape[0], :self.og_shape[1]]


class ModelBlocks(object):
    def __init__(self):
        self.weight_blocks = {}
        self.idxs = []
        self.block_ptrs = {}

        self.cur_w_idx = 0
        self.cur_w = None

        self.set_size = 0

    def add_weight(self, weight, idx):
        self.weight_blocks[idx] = weight
        self.idxs.append(idx)

        self.block_ptrs.update({k: (self.set_size, idx) for k in range(self.set_size, self.set_size + weight.set_size)})
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
        return {idx: self.weight_blocks[idx].numpy() for idx in self.idxs}

def split_matrix(array, nrows, ncols, get_shape=False):
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

    if get_shape:
        return ret, (
            num_x_blocks,
            num_y_blocks,
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
        ret, shape = split_vector(array, nrows, ncols, True)
        wblocks = WeightBlocks(shape[0], shape[1], name, w_idx, array.shape, True)
        for i in range(shape[1]):
            wblocks.add_block(0, i, ret[i])
        return wblocks
    else:
        ret, shape = split_matrix(array, nrows, ncols, True)
        wblocks = WeightBlocks(shape[0], shape[1], name, w_idx, array.shape, False)
        for i in range(shape[0]):
            for j in range(shape[1]):
                idx = i * shape[1] + j
                wblocks.add_block(i, j, ret[idx])
        return wblocks


def split_model(m, nrows, ncols, weight_lower_bound=32):
    ws = [(
        i,
        l.name,
        layer_weight_transformer(l),
    ) for i, l in enumerate(m.layers)
          if len(l.weights) != 0 and layer_bytes(l) >= weight_lower_bound]

    b = ModelBlocks()
    for idx, name, w in tqdm(ws, desc="Splitting model into blocks"):
        b.add_weight(split_weight(w, nrows, ncols, name, idx), idx)

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


# Test case
if __name__ == "__main__":
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