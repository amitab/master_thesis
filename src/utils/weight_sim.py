import tensorflow as tf
import numpy as np
import sys


def weight_comparision_info(w1, w2, i, j):
    w1 = w1.flatten()
    w2 = w2.flatten()
    # We should see if the smaller one matches the bigger one from
    # the start, and not anywhere in between
    k = min(len(w1), len(w2))
    a = np.array(w1[:k])
    b = np.array(w2[:k])
    diff = np.absolute(a - b)
    info = {
        'stats': {
            '<=0.1': np.count_nonzero(diff <= 0.1),
            '<=0.1 %': (np.count_nonzero(diff <= 0.1) * 1. / k) * 100,
            '<=0.01': np.count_nonzero(diff <= 0.01),
            '<=0.01 %': (np.count_nonzero(diff <= 0.01) * 1. / k) * 100,
            '<=0.001': np.count_nonzero(diff <= 0.001),
            '<=0.001 %': (np.count_nonzero(diff <= 0.001) * 1. / k) * 100,
            '<=0.0001': np.count_nonzero(diff <= 0.0001),
            '<=0.0001 %': (np.count_nonzero(diff <= 0.0001) * 1. / k) * 100,
            '<=0.00001': np.count_nonzero(diff <= 0.00001),
            '<=0.00001 %': (np.count_nonzero(diff <= 0.00001) * 1. / k) * 100,
            'total_elems_compared': k,
            'layers': (i, j,)
        },
        'diff': {
            'max': np.max(diff),
            'min': np.min(diff),
            'std': np.std(diff),
            'var': np.var(diff),
            'mean': np.mean(diff),
            'median': np.median(diff),
            'average': np.average(diff)
        },
        'w1': {
            'max': np.max(a),
            'min': np.min(a),
            'std': np.std(a),
            'var': np.var(a),
            'mean': np.mean(a),
            'median': np.median(a),
            'average': np.average(a)
        },
        'w2': {
            'max': np.max(b),
            'min': np.min(b),
            'std': np.std(b),
            'var': np.var(b),
            'mean': np.mean(b),
            'median': np.median(b),
            'average': np.average(b)
        }
    }

    # https://stackoverflow.com/questions/38708621/how-to-calculate-percentage-of-sparsity-for-a-numpy-array-matrix/38709042
    a[np.abs(a) < 0.00001] = 0
    info['w1']['sparisty<0.00001'] = 1.0 - (np.count_nonzero(a) /
                                            float(k))
    a[np.abs(a) < 0.0001] = 0
    info['w1']['sparisty<0.0001'] = 1.0 - (np.count_nonzero(a) /
                                            float(k))
    a[np.abs(a) < 0.001] = 0
    info['w1']['sparisty<0.001'] = 1.0 - (np.count_nonzero(a) /
                                            float(k))
    a[np.abs(a) < 0.01] = 0
    info['w1']['sparisty<0.01'] = 1.0 - (np.count_nonzero(a) /
                                            float(k))
    a[np.abs(a) < 0.1] = 0
    info['w1']['sparisty<0.1'] = 1.0 - (np.count_nonzero(a) / float(k))

    b[np.abs(b) < 0.00001] = 0
    info['w2']['sparisty<0.00001'] = 1.0 - (np.count_nonzero(b) /
                                            float(k))
    b[np.abs(b) < 0.0001] = 0
    info['w2']['sparisty<0.0001'] = 1.0 - (np.count_nonzero(b) /
                                            float(k))
    b[np.abs(b) < 0.001] = 0
    info['w2']['sparisty<0.001'] = 1.0 - (np.count_nonzero(b) /
                                            float(k))
    b[np.abs(b) < 0.01] = 0
    info['w2']['sparisty<0.01'] = 1.0 - (np.count_nonzero(b) /
                                            float(k))
    b[np.abs(b) < 0.1] = 0
    info['w2']['sparisty<0.1'] = 1.0 - (np.count_nonzero(b) / float(k))

    return info


def weight_sim(weights1, weights2, cross=True):
    sims = {}

    if cross:
        for i, w1 in enumerate(weights1):
            if w1 is None:
                continue

            if i not in sims:
                sims[i] = {}

            for j, w2 in enumerate(weights2):
                if w2 is None:
                    continue

                sims[i][j] = weight_comparision_info(w1, w2, i, j)
    else:
        # We will compare only the weights in the same indexes
        k = min(len(weights1), len(weights2))
        for i in range(k):
            w1 = weights1[i]
            w2 = weights2[i]

            sims[i] = {i: weight_comparision_info(w1, w2, i, i)}

    return sims

if __name__ == '__main__':
    model1_dir = sys.argv[1]
    model2_dir = sys.argv[2]

    model1 = tf.keras.models.load_model(model1_dir)
    model2 = tf.keras.models.load_model(model2_dir)

    weights1 = [l.get_weights()[0] if len(l.get_weights()) != 0 else None for l in model1.layers]
    weights2 = [l.get_weights()[0] if len(l.get_weights()) != 0 else None for l in model2.layers]

    sims = weight_sim(weights1, weights2)

    import pandas as pd
    df = pd.DataFrame.from_dict({(i,j): {(k, l) : sims[i][j][k][l] for k in sims[i][j].keys() for l in sims[i][j][k].keys()} for i in sims.keys() for j in sims[i].keys()}, orient='index')

    df.to_csv('test.csv')