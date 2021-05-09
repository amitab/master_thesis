import numpy as np

from matrix_utils import split, split_model, dedup_blocks
from block_compare import compare_block_sets, compare_lsh_block_sets
from tf_layer_helpers import layer_weight_transformer, layer_bytes

from tqdm import tqdm

import copy

def analyse_models(m1,
                   m2,
                   thresholds,
                   bx,
                   by,
                   weight_lower_bound=32,
                   bits=None):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    # all matrices get transformed to <= 2 dimensions
    # right now only conv2d layers can change to 2d
    weights1 = [
        layer_weight_transformer(l) for l in m1.layers
        if len(l.weights) != 0 and layer_bytes(l) >= weight_lower_bound
    ]
    weights2 = [
        layer_weight_transformer(l) for l in m2.layers
        if len(l.weights) != 0 and layer_bytes(l) >= weight_lower_bound
    ]

    assert len(weights1) > 0 and len(weights2) > 0

    s1 = []
    s2 = []

    for w1 in tqdm(weights1, desc="Splitting model 1 into blocks"):
        s1.extend(split(w1, bx, by))
    for w2 in tqdm(weights2, desc="Splitting model 2 into blocks"):
        s2.extend(split(w2, bx, by))

    if 'sim' in thresholds:
        # Compare blocks, naive
        sim_thresholds = thresholds['sim']
        print("Comparing block sets...")
        return compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)
    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        dims = bx * by
        assert bits != None
        print("Comparing LSH...")
        return compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)


def analyse_models_v2(m1,
                   m2,
                   thresholds,
                   bx,
                   by,
                   weight_lower_bound=32,
                   bits=None):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    s1 = split_model(m1, bx, by, weight_lower_bound)
    s2 = split_model(m2, bx, by, weight_lower_bound)

    if 'sim' in thresholds:
        # Compare blocks, naive
        sim_thresholds = thresholds['sim']
        print("Comparing block sets...")
        return compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)
    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        dims = bx * by
        assert bits != None
        print("Comparing LSH...")
        return compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)


def analyse_models_v2_and_dedup(m1,
                   m2,
                   thresholds,
                   bx,
                   by,
                   m1_test_fn,
                   m2_test_fn,
                   weight_lower_bound=32,
                   bits=None):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    s1 = split_model(m1, bx, by, weight_lower_bound)
    s2 = split_model(m2, bx, by, weight_lower_bound)

    m1_acc = m1_test_fn(m1)
    m2_acc = m2_test_fn(m2)

    if 'sim' in thresholds:
        # Compare blocks, naive
        sim_thresholds = thresholds['sim']
        print("Comparing block sets...")
        analysis = compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)

        for f in fp_thresholds:
            for t in sim_thresholds:
                dm1, dm2 = copy.deepcopy(m1), copy.deepcopy(m2)
                d1, d2 = dedup_blocks(analysis[f][t]['mappings'], s1, s2)
                r1 = d1.reconstruct()
                for r in r1:
                    w = m1.layers[r].weights
                    w[0] = r1[r]
                    dm1.layers[r].set_weights(w)
                r2 = d2.reconstruct()
                for r in r2:
                    w = m2.layers[r].weights
                    w[0] = r2[r]
                    dm2.layers[r].set_weights(w)

                analysis[f][t]['m1_og_acc'] = m1_acc
                analysis[f][t]['m2_og_acc'] = m2_acc
                analysis[f][t]['m1_de_acc'] = m1_test_fn(dm1)
                analysis[f][t]['m2_de_acc'] = m1_test_fn(dm2)

                del dm1
                del dm2
                del d1
                del d2


    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        dims = bx * by
        assert bits != None
        print("Comparing LSH...")
        analysis = compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)