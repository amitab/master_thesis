from deduplicator import verify_dedup_blocks
from deduplicator import analyse_models_v2_and_dedup, split_model, split, split_weight
from deduplicator import _comp_db, _comp_mem, _comp_og, _comp_mp
from deduplicator import layer_weight_transformer, layer_bytes

import tensorflow as tf
import numpy as np

from tqdm import tqdm

def test_deduplicate_models():
    TEST_MODELS_PATH = "test/"

    m1 = tf.keras.models.load_model(TEST_MODELS_PATH + 'based_model-45')
    m2 = tf.keras.models.load_model(TEST_MODELS_PATH + '30k_normal_added_10k_mix-45')

    m1._name = "based_model"
    m2._name = "30k_normal_added_10k_mix"

    fps = [0.01]
    sim = [.7]

    bx = 25
    by = 25
    weight_lower_bound = 0.1

    stats = analyse_models_v2_and_dedup(
        m1, m2,
        {
            'pairwise': {
                'fp': fps,  # for different floating point thresholds
                'sim': sim,  # for naive diff similarity percentage
            }
        },
        bx,
        by,
        weight_lower_bound,
        TEST_MODELS_PATH
    )

    m1 = tf.keras.models.load_model(TEST_MODELS_PATH + 'based_model-45')
    m2 = tf.keras.models.load_model(TEST_MODELS_PATH + '30k_normal_added_10k_mix-45')
    m3 = tf.keras.models.load_model(TEST_MODELS_PATH + 'models/based_model_0.01_0.7_0.1_25_25')
    m4 = tf.keras.models.load_model(TEST_MODELS_PATH + 'models/30k_normal_added_10k_mix_0.01_0.7_0.1_25_25')

    s1 = split_model(m1, bx, by, weight_lower_bound)
    s2 = split_model(m2, bx, by, weight_lower_bound)
    s3 = split_model(m3, bx, by, weight_lower_bound)
    s4 = split_model(m4, bx, by, weight_lower_bound)

    for f in fps:
        for t in sim:
            verify_dedup_blocks(stats[f][t]['mappings'], s3, s4, s1, s2)

def model_comp_test():
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
        assert a[0] == b[0] == c[0] == d[0]
        assert a[1] == b[1] == c[1] == d[1]
        assert {k: set(v) for k, v in a[2].items()} == \
               {k: set(v) for k, v in b[2].items()} == \
               {k: set(v) for k, v in c[2].items()} == \
               {k: set(v) for k, v in d[2].items()}
        # if not (a == b == c == d):
        #     import pdb
        #     pdb.set_trace()
        # assert a == b == c == d

    # for c, d in zip(
    #     _comp_mp(s1, s2, [0.7, 0.8, 0.9], [0.01, 0.001]),
    #     _comp_og(s1, s2, [0.7, 0.8, 0.9], [0.01, 0.001]),
    # ):
    #     assert {k: set(v) for k, v in c[2].items()} == {k: set(v) for k, v in d[2].items()}
    #     if not (c == d):
    #         test = [k  for k in c[2] if c[2][k] != d[2][k]]
    #         import pdb
    #         pdb.set_trace()
    #     assert c == d



def test_iteration():
    print("Running iteration test")
    m = tf.keras.applications.VGG16()
    weight_lower_bound = 32
    bx = 500
    by = 500

    weights = [
        layer_weight_transformer(l) for l in m.layers
        if len(l.weights) != 0 and layer_bytes(l) >= weight_lower_bound
    ]

    assert len(weights) > 0

    sa = []
    for w in tqdm(weights, desc="Splitting model into blocks"):
        sa.extend(split(w, bx, by))

    sb = split_model(m, bx, by, weight_lower_bound)

    assert len(sa) == len(sb)
    for i in range(len(sa)):
        assert np.array_equal(sa[i], sb[i])


def test_split():
    print("Running model split test")
    m = tf.keras.applications.VGG16()
    weight_lower_bound = 16
    bx = 500
    by = 500

    sb = split_model(m, bx, by, weight_lower_bound)

    for k, re_w in tqdm(sb.reconstruct(), desc="Comparing reconstructed blocks"):
        w = layer_weight_transformer(m.layers[k])
        if not np.array_equal(w, re_w):
            import pdb
            pdb.set_trace()
        assert np.array_equal(w, re_w)


def test_split_even():
    print("Running model split even test")
    x = np.random.rand(3,3,512,512)
    dims = x.shape
    s = split_weight(x.reshape(dims[0], dims[1] * dims[2] * dims[3]), 500, 500, "test", 1)

    k = s.numpy().reshape(dims)
    assert np.array_equal(x, k)

    x = np.random.rand(3,393216)
    dims = x.shape
    s = split_weight(x, 1200, 1200, "test", 1)
    k = s.numpy().reshape(dims)
    assert np.array_equal(x, k)

    x = np.random.rand(3,786432)
    dims = x.shape
    s = split_weight(x, 1200, 1200, "test", 1)
    k = s.numpy().reshape(dims)
    assert np.array_equal(x, k)

    x = np.random.rand(100,100)
    dims = x.shape
    s = split_weight(x, 1200, 1200, "test", 1)
    k = s.numpy().reshape(dims)
    assert np.array_equal(x, k)

# test_iteration()
# test_split()
# test_split_even()
# test_deduplicate_models()
model_comp_test()