import tensorflow as tf
import numpy as np
import sys


def weight_sim(model1, model2):
    sims = {}
    for i, layer1 in enumerate(model1.layers):
        # sim = []
        if len(layer1.get_weights()) == 0:
            continue

        if i not in sims:
            sims[i] = {}

        w1 = layer1.get_weights()[0].flatten()
        for j, layer2 in enumerate(model2.layers):
            if len(layer2.get_weights()) == 0:
                continue

            w2 = layer2.get_weights()[0].flatten()

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

            sims[i][j] = info
        #   sim.append(info)

        # sims.append(sim)

    return sims

if __name__ == '__main__':
    model1_dir = sys.argv[1]
    model2_dir = sys.argv[2]

    model1 = tf.keras.models.load_model(model1_dir)
    model2 = tf.keras.models.load_model(model2_dir)

    sims = weight_sim(model1, model2)

    import pandas as pd
    df = pd.DataFrame.from_dict({(i,j): {(k, l) : sims[i][j][k][l] for k in sims[i][j].keys() for l in sims[i][j][k].keys()} for i in sims.keys() for j in sims[i].keys()}, orient='index')

    df.to_csv('test.csv')