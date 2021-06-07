import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from os import listdir
from os.path import isfile, join

from tqdm import tqdm
from itertools import product

from deduplicator import evaluation, analyse_models_v2_and_dedup, CACHE_PATH


DATA_DRIFT_MODELS_PATH = "../drift/Concept Drift (Data)/"

m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + 'based_model/based_model-45')
m1._name = "based_model"
m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + '30k_normal_added_10k_mix/30k_normal_added_10k_mix-45')
m2._name = "30k_normal_added_10k_mix"

cache_files = [file_name for file_name in listdir(CACHE_PATH) if isfile(join(CACHE_PATH, file_name)) and file_name.endswith(".p") and 'l2lsh' in file_name]

for file_name in cache_files:
    info = pickle.load(open(join(CACHE_PATH, file_name), "rb"))

    d = file_name.rsplit("_", 5)
    wb = float(d[4])
    bx = int(d[2])
    by = int(d[3])

    fps = [0.01, 0.001]
    sims = [0.7, 0.8, 0.9]

    truth_stats = analyse_models_v2_and_dedup(
        m1, m2,
        {
            'pairwise': {
                'fp': fps,  # for different floating point thresholds
                'sim': sims,  # for naive diff similarity percentage
            },
        },
        bx,
        by,
        wb
    )

    # import pdb
    # pdb.set_trace()

    rs = list(info['data'].keys())
    ks = list(info['data'][rs[0]].keys())
    ls = list(info['data'][rs[0]][ks[0]].keys())
    threshold_multiplier = 1
    res = {r: {k: {l: {d: {f: {t: {} for t in sims} for f in fps} for d in range(0, l * threshold_multiplier + 1)} for l in ls} for k in ks} for r in rs}

    parameter_combination = list(product(rs, ks, ls))
    for params in tqdm(parameter_combination, desc="Running L2LSH on blocks"):
        r = params[0]
        k = params[1]
        l = params[2]

        for d in range(0, l * threshold_multiplier + 1):
            lsh_res = info['data'][r][k][l][d]['duplicate_list']
            test = [k for k in lsh_res.keys() if lsh_res[k] is None]
            lsh_reduced = info['data'][r][k][l][d]['num_reduced']

            for f in fps:
                for t in sims:
                    # import pdb
                    # pdb.set_trace()
                    euc_reduced = truth_stats['data'][f][t]['num_reduced']
                    truth_res = truth_stats['data'][f][t]['duplicate_list']
                    keys = set(list(truth_res.keys())).intersection(set(list(lsh_res.keys())))
                    keys = [k for k in keys if len(truth_res[k]) > 0]

                    truth_res = {k: v for k, v in truth_res.items() if k in keys}
                    lsh_res = {k: v for k, v in lsh_res.items() if k in keys}

                    metrics = evaluation(truth_res, lsh_res)
                    res[r][k][l][d][f][t] = metrics.tolist() + [euc_reduced, lsh_reduced]

    df = pd.DataFrame.from_dict({
        (i, j, k, l, m, n): tuple([i, j, k, l, m, n] + res[i][j][k][l][m][n])
        for i in res.keys()
        for j in res[i].keys()
        for k in res[i][j].keys()
        for l in res[i][j][k].keys()
        for m in res[i][j][k][l].keys()
        for n in res[i][j][k][l][m].keys()
    }, orient='index')

    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    df.rename(columns={
        0: 'r',
        1: 'k',
        2: 'l',
        3: 'd',
        4: 'f',
        5: 't',
        6: 'precision',
        7: 'recall',
        8: 'f1score',
        9: 'f0.5score',
        10: 'actual duplicates',
        11: 'found duplicates'
    }, inplace=True)
    df.sort_values(by=['f'], inplace=True)
    df.to_csv(f'{file_name}_analysis.csv', index=False)