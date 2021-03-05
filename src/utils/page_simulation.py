import tensorflow as tf
import numpy as np
import sys
import math


# Memory Fusion from Multiple Neural Networks
def conv2d_transformer(w):
    dims = w.shape
    assert len(dims) == 4

    return w.reshape(dims[0], dims[1] * dims[2] * dims[3])


layer_weights_transformer_mapping = {'conv2d': conv2d_transformer}


def layer_weight_transformer(layer):
    if layer.name.startswith('conv2d'):
        return layer_weights_transformer_mapping['conv2d'](
            layer.get_weights()[0])
    return layer.get_weights()[0]


def split(array, nrows, ncols):
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
    return [np.array(np.hsplit(row, num_y_blocks)) for row in rows]


def gather_blocks_to_pages(splits, num_elem_per_page):
    blocks = np.concatenate(splits)
    pages = []
    i = 0

    while i < len(blocks):
        count = 0
        page = []
        while i < len(blocks) and \
            count + blocks[i].shape[0] * blocks[i].shape[1] <= num_elem_per_page:
            page.append(blocks[i])
            count += blocks[i].shape[0] * blocks[i].shape[1]
            i += 1
        pages.append(page)
        print("Adding {} elems to page {}".format(count, len(pages) - 1))
    return pages


def model_to_pages(model, nrows, ncols, num_elem_per_page, flatten=False):
    weights = [] if flatten else {}
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) == 0:
            continue

        # We want to transform the weight to a 1D or 2D array
        w = layer_weight_transformer(layer)

        blocks = split(w, nrows, ncols)
        if flatten:
            weights.extend(gather_blocks_to_pages(blocks, num_elem_per_page))
        else:
            weights[i] = gather_blocks_to_pages(blocks, num_elem_per_page)

    return weights


if __name__ == '__main__':
    model1_dir = sys.argv[1]
    model2_dir = sys.argv[2]
    nrows = int(sys.argv[3])
    ncols = int(sys.argv[4])
    num_elem_per_page = int(sys.argv[5])
    flatten = sys.argv[6].tolower() in ('t', 'y', '1')

    model1 = tf.keras.models.load_model(model1_dir)
    model2 = tf.keras.models.load_model(model2_dir)

    pages1 = model_to_pages(model1, nrows, ncols, num_elem_per_page, flatten)
    pages2 = model_to_pages(model2, nrows, ncols, num_elem_per_page, flatten)