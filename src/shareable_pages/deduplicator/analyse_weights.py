import numpy as np

from .matrix_utils import split_matrix
from .block_compare import compare_block_sets, compare_lsh_block_sets

def analyse_weights(w1, w2, thresholds, bx, by, bits=None):
    assert len(w1.shape) == len(w2.shape) == 2

    print(f'W1 shape: {w1.shape[0]} x {w1.shape[1]}')
    print(f'W2 shape: {w2.shape[0]} x {w2.shape[1]}')

    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    a = min(w1.shape[0], w2.shape[0])
    b = min(w1.shape[1], w2.shape[1])

    diff = np.absolute(w1[:a, :b] - w2[:a, :b])
    for f in fp_thresholds:
        d = np.count_nonzero(diff <= f)
        p = (d / (a * b)) * 100
        print(f"For fp_threshold {f}, similarity = {p}%")

    print("Starting Splitting")
    s1 = split_matrix(w1, bx, by)
    print("Split w1")
    s2 = split_matrix(w2, bx, by)
    print("Split w2")

    if 'sim' in thresholds:
        # Compare blocks, naive
        sim_thresholds = thresholds['sim']
        print("Comparing block sets")
        return compare_block_sets(s1, s2, sim_thresholds, fp_thresholds)
    elif 'diff' in thresholds:
        # LSH comparision
        diff_thresholds = thresholds['diff']
        assert bits != None
        print("Comparing LSH")
        return compare_lsh_block_sets(s1, s2, diff_thresholds, bx * by, bits)