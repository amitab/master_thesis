import numpy as np

import sys
import pickle

from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm
from .utils import evaluation

from itertools import product

cache_files = [file_name for file_name in listdir("./cache") if isfile(join("./cache/", file_name)) and file_name.endswith(".p") and 'l2lsh' in file_name]
pbar = tqdm(total=len(cache_files))
for file_name in cache_files:
    # import pdb
    # pdb.set_trace()
    info = pickle.load(open(join("./cache/", file_name), "rb"))
    d = file_name.rsplit("_", 5)
    weight_lower_bound = float(d[3])


    depth = 0
    d = info['data']['eval']
    n = None
    while True:
        if type(d) == dict:
            n = list(d.keys())[0]
            d = d[n]
            depth += 1
        else:
            break

    if depth > 2:
        continue

    fps = list(info['data']['truth'].keys())
    sims = list(info['data']['truth'][fps[0]].keys())
    rs = list(info['data']['lsh'].keys())
    ks = list(info['data']['lsh'][rs[0]].keys())
    ls = list(info['data']['lsh'][rs[0]][ks[0]].keys())
    res = {r: {k: {l: {f: {t: {} for t in sims} for f in fps} for l in ls} for k in ks} for r in rs}

    parameter_combination = list(product(rs, ks, ls))
    for params in tqdm(parameter_combination, desc="Running L2LSH on blocks"):
        r = params[0]
        k = params[1]
        l = params[2]

        lsh_res = info['data']['lsh'][r][k][l]['mappings']
        for f in fps:
            for t in sims:
                truth = info['data']['truth'][f][t]['mappings']
                metrics = evaluation(truth, lsh_res)
                res[r][k][l][f][t] = metrics

    info['data']['eval'] = res

    pickle.dump(info, open(join("./cache/", file_name), "wb"))
    pbar.update(1)