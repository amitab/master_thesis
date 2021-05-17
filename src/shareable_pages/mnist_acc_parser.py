import sys
import pickle

from os import listdir
from os.path import isfile, join

import pandas as pd

# data = []
analysis = {}

ground_truth = {
    '30k_normal_added_10k_mix': {
        'MNIST': 0.9919,
        'MNIST Noisy': 0.9680
    },
    'based_model': {
        'MNIST': 0.9962,
        'MNIST Noisy': 0.1158
    }
}

fp_thresholds = []
block_sizes = []
og_size = False

cache_files = [f for f in listdir("./cache") if isfile(join("./cache/", f))]
for f in cache_files:
    d = f.rsplit("_", 5)
    block_size = tuple([d[1], d[2]])
    info = pickle.load(open(join("./cache/", f), "rb"))
    for f in info:
        og_size = False
        fp_thresholds.append(f)
        for t in info[f]:
            total_size = info[f][t]['total_blocks'] * int(block_size[0]) * int(block_size[0]) * 8
            saved_size = info[f][t]['num_reduced'] * int(block_size[0]) * int(block_size[0]) * 8
            new_size = info[f][t]['num_unique'] * int(block_size[0]) * int(block_size[0]) * 8
            info[f][t][f'Reduced Size (MB) (similarity={int(float(t) * 100)}%)'] = float(saved_size / 1e+6)
            info[f][t][f'% Reduced Size (MB) (similarity={int(float(t) * 100)}%)'] = float( (total_size - new_size) / total_size) * 100
            if not og_size:
                info[f][t][f'Original Size (MB)'] = float(total_size / 1e+6)
                info[f][t][f'Total Blocks'] = info[f][t]['total_blocks']
                og_size = True
            info[f][t][f'New Size (MB) (similarity={int(float(t) * 100)}%)'] = float(new_size / 1e+6)

            info[f][t][f'Unique Blocks (similarity={int(float(t) * 100)}%)'] = info[f][t]['num_unique']
            info[f][t][f'Duplicate Blocks (similarity={int(float(t) * 100)}%)'] = info[f][t]['num_reduced']


            del info[f][t]['mappings']
            del info[f][t]['total_blocks']
            del info[f][t]['num_unique']
            del info[f][t]['num_reduced']
            del info[f][t]['total_bytes']
            del info[f][t]['bytes_reduced']

    analysis[int(block_size[0])] = info
    block_sizes.append(int(block_size[0]))

f = open(sys.argv[1], 'r')
line = f.readline()
while line:
    if line.startswith("Model:"):
        model_info = line.split(", ")
        model_info = [i.strip() for i in model_info]
        model_name = model_info[0].split(": ")[-1]
        weight_lower_bound = model_info[1].split(" >= ")[-1]
        block_size = tuple(k.strip() for k in model_info[2].split(": ")[-1].split(" x "))
        fp_threshold = model_info[3].split(": ")[-1]
        sim_threshold = model_info[4].split(": ")[-1]

        line = f.readline()
        while len(set(line)) != 1:
            if not line:
                break
            if line.startswith("Accuracy -"):
                ra = line.split(":")
                dataset_name = ra[0].split(" - ")[-1].strip()
                model_acc = ra[-1].strip()[1:-1]
                model_acc = [float(k.strip()) for k in model_acc.split(", ")]
                loss, acc = model_acc

                try:
                    analysis[int(block_size[0])][float(fp_threshold)][float(sim_threshold)][f"{model_name.upper()} {dataset_name} accuracy reduction (similarity={int(float(sim_threshold) * 100)}%)"] = (ground_truth[model_name][dataset_name] - acc)
                except:
                    import pdb
                    pdb.set_trace()
                break

            line = f.readline()

    line = f.readline()

data = {f: {b: None for b in block_sizes} for f in fp_thresholds}

for b in analysis:
    for f in analysis[b]:
        data[f][b] = analysis[b][f]

# import pdb
# pdb.set_trace()

df = pd.DataFrame.from_dict({
    (i,j): {
        (m, n): data[i][j][m][n]
        for m in data[i][j].keys()
        for n in data[i][j][m].keys()
    }
    for i in data.keys()
    for j in sorted(data[i].keys())
}, orient='index')

df.to_csv('analysis_mnist_cnn.csv')