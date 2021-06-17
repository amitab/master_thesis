import numpy as np
import pandas as pd

from .matrix_utils import split, split_model, dedup_blocks
from .block_compare import compare_block_sets, compare_lsh_block_sets, compare_l2lsh_block_sets
from .tf_layer_helpers import layer_weight_transformer, layer_bytes

from tqdm import tqdm

import pickle
from pathlib import Path

import hashlib

import copy

import os

CACHE_PATH=f"{os.path.abspath(os.path.dirname(__file__))}/cache"

def _get_args_str(a):
    return ".".join(f"{x:.5f}" for x in a)

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


def _gather_stats(s1, s2, m1, m2, bx, by, weight_lower_bound):
    stats = {
        m1.name: {
            'total_size': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m1.layers if len(l.weights) > 0]]) * 8,
            'size_considered': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m1.layers if layer_bytes(l) >= weight_lower_bound]]) * 8,
        },
        m2.name: {
            'total_size': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m2.layers if len(l.weights) > 0]]) * 8,
            'size_considered': np.sum([np.prod(i.get_weights()[0].shape) for i in [l for l in m2.layers if layer_bytes(l) >= weight_lower_bound]]) * 8,
        },
        'data': None
    }

    return stats

def _analyse_pairwise(s1, s2, m1, m2, bx, by, save_path, weight_lower_bound, args):
    k = hashlib.md5(f"{_get_args_str(args['sim'])}_{_get_args_str(args['fp'])}".encode('utf-8')).hexdigest()
    key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{k}"
    split_cache_file = f"{CACHE_PATH}/block_analysis_{key}.p"

    stats = None
    if Path(split_cache_file).is_file():
        stats = pickle.load(open(split_cache_file, "rb"))    

    if stats is None:
        stats = _gather_stats(s1, s2, m1, m2, bx, by, weight_lower_bound)
        stats['data'] = compare_block_sets(s1, s2, args['sim'], args['fp'])
        pickle.dump(stats, open(split_cache_file, "wb"))

    m1_params = sum([sum([l.get_weights()[i].size for i in range(len(l.weights))]) for l in m1.layers if len(l.weights) > 0])
    m2_params = sum([sum([l.get_weights()[i].size for i in range(len(l.weights))]) for l in m2.layers if len(l.weights) > 0])
    analysis = stats['data']
    if save_path is not None:
        pbar = tqdm(total=len(args['fp']) * len(args['sim']), desc="Dumping deduplicated model pairs")
        for f in args['fp']:
            for t in args['sim']:
                print(f"Floating point threshold: {f}, Block similarity threshold: {t} -> Blocks ({analysis[f][t]['num_reduced']} / {analysis[f][t]['total_blocks']}) | Params ({analysis[f][t]['removed_params']} / {m1_params + m2_params}) | Padded Duplicate Blocks ({analysis[f][t]['num_padded_duplicate_blocks']} / {analysis[f][t]['num_reduced']}) | Padded Unique Blocks ({analysis[f][t]['num_padded_unique_blocks']} / {analysis[f][t]['num_unique']})")
                bak = {}
                d1, d2 = dedup_blocks(analysis[f][t]['mappings'], s1, s2)

                for idx, r in d1.reconstruct():
                    bak[idx] = copy.deepcopy(m1.layers[idx].get_weights())
                    w = m1.layers[idx].get_weights()
                    shape = w[0].shape
                    w[0] = r.reshape(*shape)
                    m1.layers[idx].set_weights(w)

                m1.save(f"{save_path}/models/{m1.name}_{f}_{t}_{weight_lower_bound}_{bx}_{by}")
                for k in bak:
                    m1.layers[k].set_weights(bak[k])
                bak.clear()

                for idx, r in d2.reconstruct():
                    bak[idx] = copy.deepcopy(m2.layers[idx].get_weights())
                    w = m2.layers[idx].get_weights()
                    shape = w[0].shape
                    w[0] = r.reshape(*shape)
                    m2.layers[idx].set_weights(w)

                m2.save(f"{save_path}/models/{m2.name}_{f}_{t}_{weight_lower_bound}_{bx}_{by}")
                for k in bak:
                    m2.layers[k].set_weights(bak[k])
                bak.clear()

                pbar.update(1)
    else:
        for f in args['fp']:
            for t in args['sim']:
                print(f"Floating point threshold: {f}, Block similarity threshold: {t} -> Blocks ({analysis[f][t]['num_reduced']} / {analysis[f][t]['total_blocks']}) | Params ({analysis[f][t]['removed_params']} / {m1_params + m2_params}) | Padded Duplicate Blocks ({analysis[f][t]['num_padded_duplicate_blocks']} / {analysis[f][t]['num_reduced']}) | Padded Unique Blocks ({analysis[f][t]['num_padded_unique_blocks']} / {analysis[f][t]['num_unique']})")

    return stats


def _analyse_cosine(s1, s2, m1, m2, bx, by, save_path, weight_lower_bound, args):
    key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{args['bits']}_{'.'.join([str(x) for x in args['diff']])}"
    split_cache_file = f"{CACHE_PATH}/cosinelsh_analysis_{key}.p"

    stats = None
    if Path(split_cache_file).is_file():
        stats = pickle.load(open(split_cache_file, "rb"))

    if stats is None:
        stats = _gather_stats(s1, s2, m1, m2, bx, by, weight_lower_bound)
        stats['data'] = compare_lsh_block_sets(s1, s2, args['diff'], bx * by, args['bits'])
        pickle.dump(stats, open(split_cache_file, "wb"))


def _analyse_l2lsh(s1, s2, m1, m2, bx, by, save_path, weight_lower_bound, args):
    k = hashlib.md5(f"{_get_args_str(args['r'])}_{_get_args_str(args['k'])}_{_get_args_str(args['l'])}_{_get_args_str([args['d']])}".encode('utf-8')).hexdigest()
    key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{k}"

    split_cache_file = f"{CACHE_PATH}/l2lsh_{key}.p"

    stats = None
    if Path(split_cache_file).is_file():
        stats = pickle.load(open(split_cache_file, "rb"))

    if stats is None:
        stats = _gather_stats(s1, s2, m1, m2, bx, by, weight_lower_bound)
        stats['data'] = compare_l2lsh_block_sets(s1, s2, args['r'], args['k'], args['l'], args['d'], bx, by)
        pickle.dump(stats, open(split_cache_file, "wb"))

    m1_params = sum([sum([l.get_weights()[i].size for i in range(len(l.weights))]) for l in m1.layers if len(l.weights) > 0])
    m2_params = sum([sum([l.get_weights()[i].size for i in range(len(l.weights))]) for l in m2.layers if len(l.weights) > 0])
    analysis = stats['data']
    if save_path is not None:
        pbar = tqdm(total=len(args['r']) * len(args['k']) * len(args['l']), desc="Dumping deduplicated model pairs")
        for r in args['r']:
            for k in args['k']:
                for l in args['l']:
                    for d in range(l * args['d']):
                        mps = analysis[r][k][l][d]
                        print(f"r: {r}, k: {k}, l: {l}, d: {d} -> Blocks ({mps['num_reduced']} / {mps['total_blocks']}) | Params ({mps['removed_params']} / {m1_params + m2_params})")

                        bak = {}
                        d1, d2 = dedup_blocks(mps['mappings'], s1, s2)

                        for idx, r in d1.reconstruct():
                            bak[idx] = m1.layers[idx].get_weights()
                            w = m1.layers[idx].get_weights()
                            shape = w[0].shape
                            w[0] = r.reshape(*shape)
                            m1.layers[idx].set_weights(w)

                        m1.save(f"{save_path}/models/{m1.name}_{r}_{k}_{l}_{weight_lower_bound}_{bx}_{by}")
                        for k in bak:
                            m1.layers[k].set_weights(bak[k])
                        bak.clear()

                        for idx, r in d2.reconstruct():
                            bak[idx] = m2.layers[idx].get_weights()
                            w = m2.layers[idx].get_weights()
                            shape = w[0].shape
                            w[0] = r.reshape(*shape)
                            m2.layers[idx].set_weights(w)

                        m2.save(f"{save_path}/models/{m2.name}_{r}_{k}_{l}_{weight_lower_bound}_{bx}_{by}")
                        for k in bak:
                            m2.layers[k].set_weights(bak[k])
                        bak.clear()

                        pbar.update(1)
    else:
        for r in args['r']:
            for k in args['k']:
                for l in args['l']:
                    for d in range(l * args['d']):
                        mps = analysis[r][k][l][d]
                        print(f"r: {r}, k: {k}, l: {l}, d: {d} -> Blocks ({mps['num_reduced']} / {mps['total_blocks']}) | Params ({mps['removed_params']} / {m1_params + m2_params})")

    return stats

"""
Takes in 2 models and splits each of the layers which have their
size >= weight_lower_bound (in terms of MB) weights into blocks
of size bx X by.
If save path is specifed, the deduplcated models will be saved in
the provided path for later inspection of accuracy.

Supports 3 methods of deduplication -
1. Pairwise - One block is compared with all other blocks.
Example:
block_set_1 = [a, b, c]
block_set_2 = [e, f, g]

a is compared to [b, c, e, f, g]
b is compared to [c, e, f, g]
c is compared to [e, f, g]
e is compared to [f, g]
f is compared to [g]

IF a ~ f,
THEN a's similarity list in the result mapping will contain f
AND f's similarity list in the result mapping will contain a

2. cosine - Doesnt work

3. l2lsh - All blocks are hashed and the similarity list for every block is
built by querying the lsh table.
"""
def analyse_models_v2_and_dedup(m1,
                   m2,
                   args,
                   bx,
                   by,
                   weight_lower_bound=32,
                   save_path=None):
    pairwise = 'pairwise' in args
    cosine = 'cosine' in args
    l2lsh = 'l2lsh' in args

    print(f"Splitting model layers with size > {weight_lower_bound} MB into blocks of size {bx} X {by} and computing similarity across blocks")
    s1 = split_model(m1, bx, by, weight_lower_bound)
    s2 = split_model(m2, bx, by, weight_lower_bound)
    assert len(s1) > 0
    assert len(s2) > 0

    if pairwise:
        stats = _analyse_pairwise(s1, s2, m1, m2, bx, by, save_path, weight_lower_bound, args['pairwise'])
    elif cosine:
        stats = _analyse_cosine(s1, s2, m1, m2, bx, by, save_path, weight_lower_bound, args['cosine'])
    elif l2lsh:
        stats = _analyse_l2lsh(s1, s2, m1, m2, bx, by, save_path, weight_lower_bound, args['l2lsh'])

    return stats
