import numpy as np

from tf_layer_helpers import layer_weight_transformer
from tqdm import tqdm


def analyse_model_weights(m1, m2, thresholds, weight_lower_bound=32):
    assert 'fp' in thresholds
    fp_thresholds = thresholds['fp']

    weights1 = [(l.name, layer_weight_transformer(l)) for l in m1.layers
                if len(l.get_weights()) != 0 and (l.count_params() * 8) /
                1e+6 >= weight_lower_bound]
    weights2 = [(l.name, layer_weight_transformer(l)) for l in m2.layers
                if len(l.get_weights()) != 0 and (l.count_params() * 8) /
                1e+6 >= weight_lower_bound]

    w_info = {}
    s1 = []
    s2 = []

    for i, x in enumerate(weights1):
        w1_name, w1 = x
        w_info[f'w1{i}'] = {'shape': w1.shape, 'name': w1_name, 'vs': {}}
        for j, y in enumerate(weights2):
            w2_name, w2 = y
            print(f"Comparing m1{i} ({w1_name}) with m2{j} ({w2_name})...")
            w_info[f'w1{i}']['vs'][f'w2{j}'] = {}
            data = w_info[f'w1{i}']['vs'][f'w2{j}']
            data['shape'] = w2.shape
            data['name'] = w2_name
            data['match'] = {}

            smol = None
            big = None
            k = None
            l = None
            if w1.flatten().shape[0] < w2.flatten().shape[0]:
                smol = w1.flatten()
                big = w2.flatten()
                k = smol.shape[0]
                l = big.shape[0]
            else:
                smol = w2.flatten()
                big = w1.flatten()
                k = smol.shape[0]
                l = big.shape[0]

            for f in fp_thresholds:
                best_match = 0
                for m in range(k, l):
                    diff = np.absolute(smol - big[m - k:m])
                    d = np.count_nonzero(diff <= f)
                    best_match = max(d, best_match)
                p = (best_match / k) * 100
                data['match'][f] = p

    return w_info