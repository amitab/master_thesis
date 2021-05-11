import numpy as np

from matrix_utils import split, split_model, dedup_blocks
from block_compare import compare_block_sets, compare_lsh_block_sets
from tf_layer_helpers import layer_weight_transformer, layer_bytes

from tqdm import tqdm

import copy
import time
import pickle
from pathlib import Path

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
                   save_path,
                   weight_lower_bound=32,
                   bits=None):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']
    sim_thresholds = thresholds.get('sim', None)
    diff_thresholds = thresholds.get('diff', None)

    block_compare = False
    print(f"Splitting model layers with size > {weight_lower_bound} MB into blocks of size {bx} X {by} and computing similarity across blocks")

    if sim_thresholds is not None:
        block_compare = True
        key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{'.'.join([str(x) for x in sim_thresholds])}_{'.'.join([str(x) for x in fp_thresholds])}"
        split_cache_file = f"cache/block_analysis_{key}.p"
    else:
        assert diff_thresholds is not None
        key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{'.'.join([str(x) for x in diff_thresholds])}_{'.'.join([str(x) for x in fp_thresholds])}"
        split_cache_file = f"cache/lsh_analysis_{key}.p"

    analysis = None
    if Path(split_cache_file).is_file():
        analysis = pickle.load(open(split_cache_file, "rb"))

    s1 = split_model(m1, bx, by, weight_lower_bound)
    s2 = split_model(m2, bx, by, weight_lower_bound)

    if block_compare:
        # Compare blocks, naive
        print("Comparing block sets...")

        if analysis is None:
            analysis = compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)
            pickle.dump(analysis, open(split_cache_file, "wb"))

        pbar = tqdm(total=len(fp_thresholds) * len(sim_thresholds), desc="Dumping deduplicated model pairs")
        for f in fp_thresholds:
            for t in sim_thresholds:
                print(f"Floating point threshold: {f}, Block similarity threshold: {t} -> Blocks ({analysis[f][t]['num_reduced']} / {analysis[f][t]['total_blocks']}) | Bytes ({analysis[f][t]['bytes_reduced']} / {analysis[f][t]['total_bytes']})")
                bak = {}
                d1, d2 = dedup_blocks(analysis[f][t]['mappings'], s1, s2)

                for idx, r in d1.reconstruct():
                    bak[idx] = m1.layers[idx].get_weights()
                    w = m1.layers[idx].get_weights()
                    w[0] = r
                    m1.layers[idx].set_weights(w)

                m1.save(f"{save_path}/models/{m1.name}_{f}_{t}_{weight_lower_bound}_{bx}_{by}")
                for k in bak:
                    m1.layers[k].set_weights(bak[k])
                bak.clear()

                for idx, r in d2.reconstruct():
                    bak[idx] = m2.layers[idx].get_weights()
                    w = m2.layers[idx].get_weights()
                    w[0] = r
                    m2.layers[idx].set_weights(w)

                m2.save(f"{save_path}/models/{m2.name}_{f}_{t}_{weight_lower_bound}_{bx}_{by}")
                for k in bak:
                    m2.layers[k].set_weights(bak[k])
                bak.clear()

                pbar.update(1)


    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        dims = bx * by
        assert bits != None
        print("Comparing LSH...")
        analysis = compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)