from __future__ import division

import math
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from lsh import signature_bit, bitcount

from enum import Enum

import gc

import sqlite3

connection = sqlite3.connect("blocks.db")
c = connection.cursor()
try:
    c.execute("DROP TABLE blocks_comp")
except:
    pass
c.execute("CREATE TABLE blocks_comp (id INT PRIMARY KEY, fp_th REAL, sim_th REAL, a_block_id TEXT, b_block_id TEXT)")

class UNIQUE_MAPPING_MODE(Enum):
    SIMILARITY = 1
    TRANSITIVE = 2


def resolve_unique_mappings(mapping,
                            len_s1,
                            len_s2,
                            num_per_block,
                            mode=UNIQUE_MAPPING_MODE.SIMILARITY):
    if mode == UNIQUE_MAPPING_MODE.TRANSITIVE:
        b_names = list(mapping.keys())

        info = {k: None for k in mapping.keys()}
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
                    if info[similar_b] is None:
                        info[similar_b] = b_name
                    temp_bs.update(mapping[similar_b])
                    del mapping[similar_b]

                pending_bs = temp_bs
                temp_bs = set()

        return {
            # 'unique_bs': unique_bs,
            'mappings': info,
            'total_blocks': (len_s1 + len_s2),
            'num_unique':
            len(unique_bs),
            'num_reduced': ((len_s1 + len_s2) - len(unique_bs)),
            'bytes_reduced':
            ((len_s1 + len_s2) - len(unique_bs)) * num_per_block * 8,
            'total_bytes': (len_s1 + len_s1) * num_per_block * 8,
        }

    elif mode == UNIQUE_MAPPING_MODE.SIMILARITY:
        info = {k: None for k in mapping}
        sorted_mappings = sorted([(k, v) for k, v in mapping.items()],
                                 key=lambda x: len(x[1]),
                                 reverse=True)

        for k, v in sorted_mappings:
            for dup in v:
                if info[dup] is None:
                    info[dup] = k

        unique_bs = [k for k in info if info[k] is None]
        unique_bs.extend([v for v in info.values() if v is not None])
        unique_bs = set(unique_bs)

        return {
            # 'unique_bs': unique_bs,
            'mappings': info,
            'total_blocks': (len_s1 + len_s2),
            'num_unique':
            len(unique_bs),
            'num_reduced': ((len_s1 + len_s2) - len(unique_bs)),
            'bytes_reduced':
            ((len_s1 + len_s2) - len(unique_bs)) * num_per_block * 8,
            'total_bytes': (len_s1 + len_s1) * num_per_block * 8,
        }


def compare_lsh_block_sets(s1, s2, diff_thresholds, dim, bits):
    ref_planes = np.random.randn(bits, dim)
    s1_lsh = [
        signature_bit(x.flatten(), ref_planes)
        for x in tqdm(s1, desc="Generating LSH signatures for set 1")
    ]
    s2_lsh = [
        signature_bit(x.flatten(), ref_planes)
        for x in tqdm(s2, desc="Generating LSH signatures for set 2")
    ]

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
                if f's2-{j}' not in info[f]:
                    info[f][f's2-{j}'] = []
                if cosine_hash <= f:
                    info[f][f's1-{i}'].append(f's2-{j}')
        pbar.update(1)

    cons = {t: {} for t in diff_thresholds}
    pbar = tqdm(total=len(diff_thresholds), desc="Gathering results")
    for f in diff_thresholds:
        cons[f] = resolve_unique_mappings(info[f], len(s1), len(s2),
                                          num_per_block)
        pbar.update(1)

    return cons

def compare_block_pairs(b1, b2, fp_thresholds, pbar):
    diff = np.absolute(b1 - b2)
    ret = [np.count_nonzero(diff <= f) for f in fp_thresholds]
    pbar.update(1)
    return ret

def _comp_db(s1, s2, sim_thresholds, fp_thresholds):
    num_per_block = s1[0].shape[0] * s1[0].shape[1]
    cursor = connection.cursor()

    def process_op(data):
        recs = []
        for a in data:
            for b in data[a]:
                for l, f in enumerate(fp_thresholds):
                    d = data[a][b][l]
                    for t in sim_thresholds:
                        if d / num_per_block >= t:
                            recs.append(tuple([f, t, a, b]))

        cursor.executemany('INSERT INTO blocks_comp(fp_th, sim_th, a_block_id, b_block_id) VALUES(?,?,?,?);', recs)

    batch_size = 5000

    s = 0
    e = batch_size
    pbar = tqdm(total=(len(s1) * (len(s1) - 1)) / 2, desc="Dedup S1")
    while s < len(s1):
        a = { f"s1-{i}":
            { f"s1-{j}":
                compare_block_pairs(s1[i], s1[j], fp_thresholds, pbar) for j in range(i + 1, len(s1))
            } for i in range(s, min(len(s1), e))
        }
        process_op(a)
        del a
        gc.collect()
        s += batch_size
        e += batch_size

    s = 0
    e = batch_size
    pbar = tqdm(total=(len(s2) * (len(s2) - 1)) / 2, desc="Dedup S2")
    while s < len(s2):
        b = { f"s2-{i}":
                { f"s2-{j}":
                    compare_block_pairs(s2[i], s2[j], fp_thresholds, pbar) for j in range(i + 1, len(s2))
                } for i in range(s, min(len(s2), e))
        }
        process_op(b)
        del b
        gc.collect()
        s += batch_size
        e += batch_size

    s = 0
    e = batch_size
    pbar = tqdm(total=len(s1) * len(s2), desc="Dedup S1 and S2")
    while s < len(s1):
        c = { f"s1-{i}":
            { f"s2-{j}":
                compare_block_pairs(s1[i], s2[j], fp_thresholds, pbar) for j in range(len(s2))
            } for i in range(s, min(len(s1), e))
        }
        process_op(c)
        del c
        gc.collect()
        s += batch_size
        e += batch_size

    for f in fp_thresholds:
        for t in sim_thresholds:
            inf = {f's{i}-{j}': None for i in range(1, 3) for j in range(len(s1) if i == 1 else len(s2))}
            e = cursor.execute("SELECT a_block_id, group_concat(b_block_id) FROM blocks_comp WHERE fp_th = ? AND sim_th = ? GROUP BY a_block_id", (f, t))
            r = dict(e.fetchall())
            inf.update(r)
            inf = {k: v.split(",") if v is not None else [] for k, v in inf.items()}

            yield f, t, inf

def _comp_mem(s1, s2, sim_thresholds, fp_thresholds):
    info = {f: {t: {} for t in sim_thresholds} for f in fp_thresholds}
    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    def process_op(data):
        for a in data:
            for b in data[a]:
                for l, f in enumerate(fp_thresholds):
                    d = data[a][b][l]
                    for t in sim_thresholds:
                        if a not in info[f][t]:
                            info[f][t][a] = []
                        if d / num_per_block >= t:
                            info[f][t][a].append(b)

    pbar = tqdm(total=(len(s1) * (len(s1) - 1)) / 2, desc="Dedup S1")
    a = { f"s1-{i}":
        { f"s1-{j}":
            compare_block_pairs(s1[i], s1[j], fp_thresholds, pbar) for j in range(i + 1, len(s1))
        } for i in range(len(s1))
    }
    process_op(a)
    pbar = tqdm(total=(len(s2) * (len(s2) - 1)) / 2, desc="Dedup S2")
    b = { f"s2-{i}":
        { f"s2-{j}":
            compare_block_pairs(s2[i], s2[j], fp_thresholds, pbar) for j in range(i + 1, len(s2))
        } for i in range(len(s2))
    }
    process_op(b)
    pbar = tqdm(total=len(s1) * len(s2), desc="Dedup S1 and S2")
    c = { f"s1-{i}":
        { f"s2-{j}":
            compare_block_pairs(s1[i], s2[j], fp_thresholds, pbar) for j in range(len(s2))
        } for i in range(len(s1))
    }
    process_op(c)

    for f in fp_thresholds:
        for t in sim_thresholds:
            info[f][t][f's2-{len(s2) - 1}'] = []
            yield f, t, info[f][t]

def _comp_mp(s1, s2, sim_thresholds, fp_thresholds):
    import multiprocessing

    info = {f: {t: {} for t in sim_thresholds} for f in fp_thresholds}
    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    ###################
    ## MULTI PROCESS ##
    ###################

    q = multiprocessing.Queue()
    pq = multiprocessing.Queue()

    def monitor(pq, t):
        pbar = tqdm(total = (len(s1) * (len(s1) - 1)) / 2 + (len(s2) * (len(s2) - 1)) / 2 + len(s1) * len(s2), desc="Deduplicating Sets")
        while pq.get() == 1:
            pbar.update(1)

    def compare_blocks(b1, b2, fp_thresholds, pq):
        diff = np.absolute(b1 - b2)
        ret = [np.count_nonzero(diff <= f) for f in fp_thresholds]
        pq.put(1)
        return ret

    def doit(a, b, c, d, q, pq):
        q.put({ f"s1-{i}":
            { f"s1-{j}":
                compare_blocks(s1[i], s1[j], fp_thresholds, pq) for j in range(i + 1, len(s1))
            } for i in range(a, b)
        })
        q.put({ f"s2-{i}":
            { f"s2-{j}":
                compare_blocks(s2[i], s2[j], fp_thresholds, pq) for j in range(i + 1, len(s2))
            } for i in range(c, d)
        })
        q.put({ f"s1-{i}":
            { f"s2-{j}":
                compare_blocks(s1[i], s2[j], fp_thresholds, pq) for j in range(len(s2))
            } for i in range(a, b)
        })

    def process_op(data):
        for a in data:
            for b in data[a]:
                for l, f in enumerate(fp_thresholds):
                    d = data[a][b][l]
                    for t in sim_thresholds:
                        if a not in info[f][t]:
                            info[f][t][a] = []
                        if d / num_per_block >= t:
                            info[f][t][a].append(b)

    num_processes = 2
    splits_s1 = [i * int(len(s1) / num_processes) for i in range(num_processes)] + [len(s1)]
    splits_s2 = [i * int(len(s2) / num_processes) for i in range(num_processes)] + [len(s2)]

    ps = [
        multiprocessing.Process(target=doit, args=(splits_s1[i], splits_s1[i+1], splits_s2[i], splits_s2[i+1], q, pq), name=1)
        for i in range(num_processes)
    ]

    mon_proc = multiprocessing.Process(target=monitor, args=(pq, 1), name=1)
    mon_proc.start()

    for p in ps:
        p.start()

    for _ in range(num_processes * 3):
        process_op(q.get())

    for f in fp_thresholds:
        for t in sim_thresholds:
            info[f][t][f's2-{len(s2) - 1}'] = []

    pq.put(None)

    q.close()
    q.join_thread()

    pq.close()
    pq.join_thread()

    for p in ps:
        p.join()
    
    mon_proc.join()

    for f in fp_thresholds:
        for t in sim_thresholds:
            info[f][t][f's2-{len(s2) - 1}'] = []
            yield f, t, info[f][t]


def _comp_og(s1, s2, sim_thresholds, fp_thresholds):
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
                    # if f's2-{j}' not in info[f][t]:
                    #     info[f][t][f's2-{j}'] = []
                    if d / tot >= t:
                        info[f][t][f's1-{i}'].append(f's2-{j}')
        pbar.update(1)

    for f in fp_thresholds:
        for t in sim_thresholds:
            info[f][t][f's2-{len(s2) - 1}'] = []
            yield f, t, info[f][t]

def compare_block_sets(s1, s2, sim_thresholds, fp_thresholds):
    info = {f: {t: {} for t in sim_thresholds} for f in fp_thresholds}
    num_per_block = s1[0].shape[0] * s1[0].shape[1]

    cons = {f: {t: 0 for t in sim_thresholds} for f in fp_thresholds}
    pbar = tqdm(total=len(fp_thresholds), desc="Gathering results")
    if len(s1) <= 500 and len(s2) <= 500:
        for k in _comp_mem(s1, s2, sim_thresholds, fp_thresholds):
            f, t, mappings = k
            temp = resolve_unique_mappings(mappings, len(s1), len(s2), num_per_block)
            cons[f][t] = temp
            pbar.update(1)
    else:
        for k in _comp_db(s1, s2, sim_thresholds, fp_thresholds):
            f, t, mappings = k
            temp = resolve_unique_mappings(mappings, len(s1), len(s2), num_per_block)
            cons[f][t] = temp
            pbar.update(1)

    return cons


# Test case
if __name__ == "__main__":
    from matrix_utils import split_model

    print("Running model comp test")
    m1 = tf.keras.applications.VGG16()
    m2 = tf.keras.applications.VGG19()
    weight_lower_bound = 16
    bx = 1500
    by = 1500

    s1 = split_model(m1, bx, by, weight_lower_bound)
    s2 = split_model(m2, bx, by, weight_lower_bound)

    for a, b, c, d in zip(
        _comp_db(s1, s2, [0.7, 0.8, 0.9], [0.01, 0.001]),
        _comp_mem(s1, s2, [0.7, 0.8, 0.9], [0.01, 0.001]),
        _comp_mp(s1, s2, [0.7, 0.8, 0.9], [0.01, 0.001]),
        _comp_og(s1, s2, [0.7, 0.8, 0.9], [0.01, 0.001]),
    ):
        assert a == b == c == d