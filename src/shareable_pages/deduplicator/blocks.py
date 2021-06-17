import math
import numpy as np


def idx_to_2d(num_y_blocks, idx):
    i = math.floor(idx / num_y_blocks)
    j = idx % num_y_blocks
    return (
        i,
        j,
    )


class Block(object):
    def __init__(self, block, i, j, num_x_blocks, num_y_blocks, name, w_idx,
                 is_vec, unpadded_shape):
        self.block = block
        self.i = i
        self.j = j
        self.num_x_blocks = num_x_blocks
        self.num_y_blocks = num_y_blocks
        self.name = name
        self.w_idx = w_idx
        self.is_vec = is_vec
        self.unpadded_shape = unpadded_shape

    def get_padded_size(self):
        return self.block.size

    def get_unpadded_size(self):
        return np.product(self.unpadded_shape)

    def is_padded(self):
        return self.get_padded_size() != self.get_unpadded_size()


class WeightBlocks(object):
    def __init__(self,
                 blocking_shape,
                 name,
                 w_idx,
                 og_shape,
                 is_vec,
                 reshape=None):
        self.num_x_blocks, self.num_y_blocks = blocking_shape
        self.name = name
        self.w_idx = w_idx
        self.blocks = {}
        # self.all_blocks = []
        self.is_vec = is_vec
        self.set_size = self.num_x_blocks * self.num_y_blocks
        self.og_shape = og_shape
        self.reshape = reshape

        self.cur_i = 0
        self.cur_j = 0

    def add_block(self, i, j, block, block_unpadded_shape=None):
        assert i not in self.blocks or j not in self.blocks[i]
        if i not in self.blocks:
            self.blocks[i] = {}

        self.blocks[i][j] = Block(block, i, j, self.num_x_blocks,
                                  self.num_y_blocks, self.name, self.w_idx,
                                  self.is_vec, block_unpadded_shape)
        # self.all_blocks.append(block)

    def __getitem__(self, idx):
        i, j = idx_to_2d(self.num_y_blocks, idx)
        return self.blocks[i][j].block

    def getBlock(self, idx):
        i, j = idx_to_2d(self.num_y_blocks, idx)
        return self.blocks[i][j]

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
            matrix = matrix.reshape(self.reshape)

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

    def getBlock(self, idx):
        start, w_idx = self.block_ptrs[idx]
        b_idx = idx - start
        return self.weight_blocks[w_idx].getBlock(b_idx)

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