import tensorflow as tf
import numpy as np
import sys
import math
import pandas as pd

import weight_sim


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
    assert len(layer.get_weights()[0].shape) <= 2
    return layer.get_weights()[0]


def split(array, nrows, ncols=None):
    if ncols is None:
        return split_vector(array, nrows)
    else:
        return split_matrix(array, nrows, ncols)


def split_vector(array, block_size):
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
    return ret


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


def gather_blocks_to_pages(blocks, num_elem_per_page):
    pages = []
    i = 0

    while i < len(blocks):
        count = 0
        page = []
        while i < len(blocks) and count + blocks[i].size <= num_elem_per_page:
            page.append(blocks[i])
            count += blocks[i].size
            i += 1
        pages.append(page)
        print("Adding {} elems to page {}".format(count, len(pages) - 1))
    return pages


def model_to_pages(model, nrows, ncols, num_elem_per_page):
    weights = {}
    for i, layer in enumerate(model.layers):
        if len(layer.get_weights()) == 0:
            continue

        # We want to transform the weight to a 1D or 2D array
        w = layer_weight_transformer(layer)

        if len(w.shape) == 1:
            blocks = split(w, nrows * ncols)
        else:
            blocks = split(w, nrows, ncols)
        weights[i] = gather_blocks_to_pages(blocks, num_elem_per_page)

    return weights


if __name__ == '__main__':
    model1_dir = sys.argv[1]
    model2_dir = sys.argv[2]
    nrows = int(sys.argv[3])
    ncols = int(sys.argv[4])
    num_elem_per_page = int(sys.argv[5])
    flatten = sys.argv[6].lower() in ('t', 'y', '1')

    assert num_elem_per_page >= nrows * ncols, "page size less than a block!"

    model1 = tf.keras.models.load_model(model1_dir)
    model2 = tf.keras.models.load_model(model2_dir)

    pages1 = model_to_pages(model1, nrows, ncols, num_elem_per_page)
    pages2 = model_to_pages(model2, nrows, ncols, num_elem_per_page)

    import pdb
    pdb.set_trace()

    # if flatten:
    #     # Compare all pages to all pages

    #     sim_info = {}

    #     for layer1 in pages1:
    #         if layer1 not in sim_info:
    #             sim_info[layer1] = {}
    #         for layer2 in pages2:
    #             if layer2 not in sim_info[layer1]:
    #                 sim_info[layer1][layer2] = {}

    #             l1_l2_sim = sim_info[layer1][layer2]
    #             for i, p1 in enumerate(pages1[layer1]):
    #                 if i not in l1_l2_sim:
    #                     l1_l2_sim[i] = {}
    #                 for j, p2 in enumerate(pages2[layer2]):
    #                     l1_l2_sim[i][j] = weight_sim.weight_comparision_info(p1, p2, i, j)
        
    #     # Now to write this to csv
    #     df = pd.DataFrame.from_dict({
    #         (i,j,k,l): {
    #             (m, n): sim_info[i][j][k][l][m][n]
    #             for m in sim_info[i][j][k][l].keys()
    #             for n in sim_info[i][j][k][l][m].keys()
    #         }
    #         for i in sim_info.keys()
    #         for j in sim_info[i].keys()
    #         for k in sim_info[i][j].keys()
    #         for l in sim_info[i][j][k].keys()
    #     }, orient='index')
    #     df.to_csv('test.csv')

    # else:
    #     # Compare the pages belonging to the same layer's weights
    #     # Makes sense only when comparing two models with same arch
        
    #     sim_info = {}

    #     if len(pages1) < len(pages2):
    #         smol_layers = pages1
    #         big_layers = pages2
    #     else:
    #         smol_layers = pages2
    #         big_layers = pages1

    #     for l in smol_layers:
    #         # This shouldnt happen if same model arch
    #         if l not in big_layers:
    #             continue

    #         if l not in sim_info:
    #             sim_info[l] = {}

    #         for i, p1 in enumerate(smol_layers[l]):
    #             if i not in sim_info[l]:
    #                 sim_info[l][i] = {}

    #             for j, p2 in enumerate(big_layers[l]):
    #                 sim_info[l][i][j] = weight_sim.weight_comparision_info(p1, p2, i, j)

    #     # Now to write this to csv
    #     df = pd.DataFrame.from_dict({
    #         (i,j,k): {
    #             (m, n): sim_info[i][j][k][m][n]
    #             for m in sim_info[i][j][k].keys()
    #             for n in sim_info[i][j][k][m].keys()
    #         }
    #         for i in sim_info.keys()
    #         for j in sim_info[i].keys()
    #         for k in sim_info[i][j].keys() 
    #     }, orient='index')
    #     df.to_csv('test.csv')