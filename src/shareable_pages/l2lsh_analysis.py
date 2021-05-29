import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
import pandas as pd
from utils import evaluation
from tqdm import tqdm
from itertools import product

from block_compare import _resolve_mappings

import tensorflow as tf
from matrix_utils import split_model
from block_compare import _comp_mem

DATA_DRIFT_MODELS_PATH = "../drift/Concept Drift (Data)/"

m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + 'based_model/based_model-45')
m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + '30k_normal_added_10k_mix/30k_normal_added_10k_mix-45')


def build_lsh(mappings):
    lsh_res = {}
    for k, v in mappings.items():
        se = []
        px, py = k.split("-")
        py = int(py)
        psetId = int(px[1])
        for val in v:
            x, y = val.split("-")
            y = int(y)
            setId = int(x[1])

            if setId >= psetId and y > py:
                se.append(val)
        lsh_res[k] = se
    return lsh_res

cache_files = [file_name for file_name in listdir("./cache") if isfile(join("./cache/", file_name)) and file_name.endswith(".p") and 'l2lsh' in file_name]

for file_name in cache_files:
    info = pickle.load(open(join("./cache/", file_name), "rb"))

##############
    d = file_name.rsplit("_", 5)
    wb = float(d[4])
    bx = int(d[2])
    by = int(d[3])

    s1 = split_model(m1, bx, by, wb)
    s2 = split_model(m2, bx, by, wb)
###############

    fps = list(info['data']['truth'].keys())
    sims = list(info['data']['truth'][fps[0]].keys())
    rs = list(info['data']['lsh'].keys())
    ks = list(info['data']['lsh'][rs[0]].keys())
    ls = list(info['data']['lsh'][rs[0]][ks[0]].keys())
    res = {r: {k: {l: {f: {t: {} for t in sims} for f in fps} for l in ls} for k in ks} for r in rs}

    truth_euc = {f: {t: {} for t in sims} for f in fps}
    for f, t, mappings in _comp_mem(s1, s2, sims, fps):
        truth_euc[f][t] = mappings

    parameter_combination = list(product(rs, ks, ls))
    for params in tqdm(parameter_combination, desc="Running L2LSH on blocks"):
        r = params[0]
        k = params[1]
        l = params[2]

#########
        lsh_res = info['data']['lsh'][r][k][l]['mappings']
        lsh_reduced = info['data']['lsh'][r][k][l]['resolved']['num_reduced']
##########
        # lsh_res = build_lsh(info['data']['lsh'][r][k][l]['mappings'])
        # uniq = _resolve_mappings(lsh_res)[1]
        # lsh_reduced = info['data']['lsh'][r][k][l]['resolved']['total_blocks'] - len(uniq)
        # if r == 1.1 and k == 17 and l == 28:
        #     import pdb
        #     pdb.set_trace()

        for f in fps:
            for t in sims:
##########
                uniq = _resolve_mappings(truth_euc[f][t])[1]
                euc_reduced = info['data']['truth'][f][t]['resolved']['total_blocks'] - len(uniq)
                # if euc_reduced != info['data']['truth'][f][t]['resolved']['num_reduced']:
                #     print(f"{info['data']['truth'][f][t]['resolved']['num_reduced']} -> {euc_reduced}")
                metrics = evaluation(truth_euc[f][t], lsh_res)
                res[r][k][l][f][t] = metrics.tolist() + [euc_reduced, lsh_reduced]
###########
                # truth = info['data']['truth'][f][t]['mappings']
                # metrics = evaluation(truth, lsh_res)
                # res[r][k][l][f][t] = metrics.tolist() + [info['data']['truth'][f][t]['resolved']['num_reduced'], lsh_reduced]

    # pickle.dump(res, open(f"{file_name}_l2lsh_analysis.p", "wb"))

    # import pdb
    # pdb.set_trace()

    df = pd.DataFrame.from_dict({
        (i, j, k, l, m): tuple([i, j, k, l, m] + res[i][j][k][l][m])
        for i in res.keys()
        for j in res[i].keys()
        for k in res[i][j].keys()
        for l in res[i][j][k].keys()
        for m in res[i][j][k][l].keys()
    }, orient='index')

    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    df.rename(columns={
        0: 'r',
        1: 'k',
        2: 'l',
        3: 'f',
        4: 't',
        5: 'precision',
        6: 'recall',
        7: 'f1score',
        8: 'f0.5score',
        9: 'actual duplicates',
        10: 'found duplicates'
    }, inplace=True)
    df.sort_values(by=['f'], inplace=True)
    df.to_csv(f'{file_name}_analysis.csv', index=False)