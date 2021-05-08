import tensorflow as tf
import numpy as np

from tqdm import tqdm

import math

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