from __future__ import division

import math
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from lsh import signature_bit, bitcount

from enum import Enum

class UNIQUE_MAPPING_MODE(Enum):
    SIMILARITY = 1
    TRANSITIVE = 2

def resolve_unique_mappings(mapping, len_s1, len_s2, num_per_block, mode=UNIQUE_MAPPING_MODE.TRANSITIVE):
    if mode == UNIQUE_MAPPING_MODE.TRANSITIVE:
        b_names = list(mapping.keys())

        unique_bs = set()

        for b_name in b_names:
            if not b_name in mapping:
                continue
            unique_bs.add(b_name)
            pending_bs = set(mapping[b_name])
            pending_bs.add(b_name)
            del mapping[b_name]
            temp_bs = set()
            while pending_bs:
                for similar_b in pending_bs:
                    if not similar_b in mapping:
                        continue
                    temp_bs.update(mapping[similar_b])
                    del mapping[similar_b]

                pending_bs = temp_bs
                temp_bs = set()

        return {
            # 'unique_bs': unique_bs,
            'total_blocks': (len_s1 + len_s2),
            'num_unique':
            len(unique_bs),
            'num_reduced': ((len_s1 + len_s2) - len(unique_bs)),
            'bytes_reduced':
            ((len_s1 + len_s2) - len(unique_bs)) * num_per_block * 8,
            'total_bytes': (len_s1 + len_s1) * num_per_block * 8,
        }    

    # elif mode == UNIQUE_MAPPING_MODE.SIMILARITY:
        



def compare_lsh_block_sets(s1, s2, diff_thresholds, dim, bits):
    ref_planes = np.random.randn(bits, dim)
    s1_lsh = [signature_bit(x.flatten(), ref_planes) for x in tqdm(s1, desc="Generating LSH signatures for set 1")]
    s2_lsh = [signature_bit(x.flatten(), ref_planes) for x in tqdm(s2, desc="Generating LSH signatures for set 2")]

    info = {t: {} for t in diff_thresholds}
    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    for i in tqdm(range(len(s1)), desc="Deduplicating Set 1"):
        sig1 = s1_lsh[i]
        for j in range(i + 1, len(s1)):
            assert s1[i].shape == s1[j].shape
            sig2 = s1_lsh[j]
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if f's1-{i}' not in info[f]:
                    info[f][f's1-{i}'] = []
                if cosine_hash <= f:
                    info[f][f's1-{i}'].append(f's1-{j}')

    for i in tqdm(range(len(s2)), desc="Deduplicating Set 2"):
        sig1 = s2_lsh[i]
        for j in range(i + 1, len(s2)):
            assert s2[i].shape == s2[j].shape
            sig2 = s2_lsh[j]
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if f's2-{i}' not in info[f]:
                    info[f][f's2-{i}'] = []
                if cosine_hash <= f:
                    info[f][f's2-{i}'].append(f's2-{j}')

    pbar = tqdm(total=len(s1), desc="Deduplicating Set 1 and Set 2")
    for i, b1 in enumerate(s1):
        sig1 = s1_lsh[i]
        for j, b2 in enumerate(s2):
            assert b1.shape == b2.shape
            sig2 = s2_lsh[j]
            cosine_hash = 1 - bitcount(sig1 ^ sig2) / bits
            for f in diff_thresholds:
                if f's1-{i}' not in info[f]:
                    info[f][f's1-{i}'] = []
                if cosine_hash <= f:
                    info[f][f's1-{i}'].append(f's2-{j}')
        pbar.update(1)

    cons = {t: {} for t in diff_thresholds}
    pbar = tqdm(total=len(diff_thresholds), desc="Gathering results")
    for f in diff_thresholds:
        cons[f] = resolve_unique_mappings(info[f], len(s1), len(s2), num_per_block)
        pbar.update(1)

    return cons


def compare_block_sets(s1, s2, sim_thresholds, fp_thresholds):
    info = {f: {t: {} for t in sim_thresholds} for f in fp_thresholds}

    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    for i in tqdm(range(len(s1)), desc="Deduplicating Set 1"):
        for j in range(i + 1, len(s1)):
            assert s1[i].shape == s1[j].shape
            diff = np.absolute(s1[i] - s1[j])
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = s1[i].shape[0] * s1[i].shape[1]
                for t in sim_thresholds:
                    if f's1-{i}' not in info[f][t]:
                        info[f][t][f's1-{i}'] = []
                    if d / tot >= t:
                        info[f][t][f's1-{i}'].append(f's1-{j}')

    for i in tqdm(range(len(s2)), desc="Deduplicating Set 2"):
        for j in range(i + 1, len(s2)):
            assert s2[i].shape == s2[j].shape
            diff = np.absolute(s2[i] - s2[j])
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = s2[i].shape[0] * s2[i].shape[1]
                for t in sim_thresholds:
                    if f's2-{i}' not in info[f][t]:
                        info[f][t][f's2-{i}'] = []
                    if d / tot >= t:
                        info[f][t][f's2-{i}'].append(f's2-{j}')

    pbar = tqdm(total=len(s1), desc="Deduplicating Set 1 and Set 2")
    for i, b1 in enumerate(s1):
        for j, b2 in enumerate(s2):
            assert b1.shape == b2.shape
            diff = np.absolute(b1 - b2)
            for f in fp_thresholds:
                d = np.count_nonzero(diff <= f)
                tot = b1.shape[0] * b2.shape[1]
                for t in sim_thresholds:
                    if f's1-{i}' not in info[f][t]:
                        info[f][t][f's1-{i}'] = []
                    if d / tot >= t:
                        info[f][t][f's1-{i}'].append(f's2-{j}')
        pbar.update(1)

    cons = {f: {t: 0 for t in sim_thresholds} for f in fp_thresholds}
    pbar = tqdm(total=len(fp_thresholds), desc="Gathering results")
    for f in fp_thresholds:
        for t in sim_thresholds:
            cons[f][t] = resolve_unique_mappings(info[f][t], len(s1), len(s2), num_per_block)
        pbar.update(1)

    return cons
