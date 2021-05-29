import numpy as np

from .matrix_utils import split, split_model, dedup_blocks
from .block_compare import compare_block_sets, compare_lsh_block_sets, compare_l2lsh_block_sets
from .tf_layer_helpers import layer_weight_transformer, layer_bytes

from tqdm import tqdm

import copy
import time
import pickle
from pathlib import Path

CACHE_PATH="./cache"

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
    key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{'.'.join([str(x) for x in args['sim']])}_{'.'.join([str(x) for x in args['fp']])}"
    split_cache_file = f"{CACHE_PATH}/block_analysis_{key}.p"

    stats = None
    if Path(split_cache_file).is_file():
        stats = pickle.load(open(split_cache_file, "rb"))    

    if stats is None:
        stats = _gather_stats(s1, s2, m1, m2, bx, by, weight_lower_bound)
        stats['data'] = compare_block_sets(s1, s2, args['sim'], args['fp'])
        pickle.dump(stats, open(split_cache_file, "wb"))

    import pdb
    pdb.set_trace()

    analysis = stats['data']
    if save_path is not None:
        pbar = tqdm(total=len(args['fp']) * len(args['sim']), desc="Dumping deduplicated model pairs")
        for f in args['fp']:
            for t in args['sim']:
                print(f"Floating point threshold: {f}, Block similarity threshold: {t} -> Blocks ({analysis[f][t]['num_reduced']} / {analysis[f][t]['total_blocks']}) | Bytes ({analysis[f][t]['bytes_reduced']} / {analysis[f][t]['total_bytes']})")
                bak = {}
                d1, d2 = dedup_blocks(analysis[f][t]['mappings'], s1, s2)

                for idx, r in d1.reconstruct():
                    bak[idx] = m1.layers[idx].get_weights()
                    w = m1.layers[idx].get_weights()
                    shape = w[0].shape
                    w[0] = r.reshape(*shape)
                    m1.layers[idx].set_weights(w)

                m1.save(f"{save_path}/models/{m1.name}_{f}_{t}_{weight_lower_bound}_{bx}_{by}")
                for k in bak:
                    m1.layers[k].set_weights(bak[k])
                bak.clear()

                for idx, r in d2.reconstruct():
                    bak[idx] = m2.layers[idx].get_weights()
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
                print(f"Floating point threshold: {f}, Block similarity threshold: {t} -> Blocks ({analysis[f][t]['num_reduced']} / {analysis[f][t]['total_blocks']}) | Bytes ({analysis[f][t]['bytes_reduced']} / {analysis[f][t]['total_bytes']})")


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
    k = f"{'.'.join([str(x) for x in args['r']])}_{'.'.join([str(x) for x in args['k']])}_{'.'.join([str(x) for x in args['l']])}_{'.'.join([str(x) for x in args['sims']])}_{'.'.join([str(x) for x in args['fps']])}"
    key = f"{m1.name}_{m2.name}_{bx}_{by}_{weight_lower_bound}_{hash(k)}"
    split_cache_file = f"{CACHE_PATH}/l2lsh_{key}.p"

    stats = None
    if Path(split_cache_file).is_file():
        stats = pickle.load(open(split_cache_file, "rb"))

    if stats is None:
        stats = _gather_stats(s1, s2, m1, m2, bx, by, weight_lower_bound)
        stats['data'] = compare_l2lsh_block_sets(s1, s2, args['r'], args['k'], args['l'], args['fps'], args['sims'], bx, by)
        pickle.dump(stats, open(split_cache_file, "wb"))

    analysis = stats['data']['lsh']
    if save_path is not None:
        pbar = tqdm(total=len(args['r']) * len(args['k']) * len(args['l']), desc="Dumping deduplicated model pairs")
        for r in args['r']:
            for k in args['k']:
                for l in args['l']:
                    mps = analysis[r][k][l]['resolved']

                    print(f"r: {r}, k: {k}, l: {l} -> Blocks ({mps['num_reduced']} / {mps['total_blocks']}) | Bytes ({mps['bytes_reduced']} / {mps['total_bytes']})")
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
                    mps = analysis[r][k][l]['resolved']
                    print(f"r: {r}, k: {k}, l: {l} -> Blocks ({mps['num_reduced']} / {mps['total_blocks']}) | Bytes ({mps['bytes_reduced']} / {mps['total_bytes']})")

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