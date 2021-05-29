## Start from main.py

`analyse_models_v2_and_dedup` is the function used to deduplicate two models.
It takes in 2 models and splits each of the layers which have their size >= `weight_lower_bound` (in terms of MB) weights into blocks of size bx X by. If save path is specifed, the deduplcated models will be saved in the provided path for later inspection of accuracy.

The `weight_lower_bound` is important since we dont care about deduplicating smaller weights which occupy less size on disk. In a database, pages are usually in sizes of 16, 32, 64 MB. 

A cache file is written to disk with the deduplication information. This can be reused to reconstruct the models without waiting for building deduplication information again.

Supports 3 methods of deduplication -
1. Pairwise - One block is compared with all other blocks.
IF a ~ f, THEN a's similarity list in the result mapping will contain f AND f's similarity list in the result mapping will contain a
Example:
block_set_1 = [a, b, c]
block_set_2 = [e, f, g]
    1. a is compared to [b, c, e, f, g]
    1. b is compared to [c, e, f, g]
    1. c is compared to [e, f, g]
    1. e is compared to [f, g]
    1. f is compared to [g]
1. cosine - Doesnt work
1. l2lsh - All blocks are hashed and the similarity list for every block is built by querying the lsh table.

A model's weights are split into blocks. The metadata for each block (the block coordinates and the layer it belongs to) are maintained, so that the entire model can be reconstructed again. The metadata information is maintained using classes defined in `deduplicator/blocks.py`. `ModelBlocks` class contains the weight information, and the layers these weights belong to. `WeightBlocks` class contains the original dimensions of the weight (since blocking evenly requires padding the matrix) and the blocks that belong to it. `Block` class contains other misc metadata and the block (of type `numpy.ndarray`) itself.

Currently this works only with Tensorflow v2 models.

## Pairwise Comparision

For pairwise comparision, the code path goes from `analyse_models_v2_and_dedup` -> `_analyse_pairwise` -> `compare_block_sets` -> either `_comp_mem` or `_comp_db` (depending on how many blocks are being compared. Uses sqlite to store temporary results so that the python process doesnt killed due to OOM).

Any of the `_comp_(mem|db)` does the actual blocks comparision.

## L2LSH Comparision

For L2LSH comparision, the code path goes from `analyse_models_v2_and_dedup` -> `_analyse_l2lsh` -> `compare_l2lsh_block_sets` -> `_compare_l2lsh`. Beware, depending on the number of parameters provided, this can take a long time. The cache file dumped may have a lot of data too.

Right now, the cache file cannot be reused because the hash generated to identify a unique set of parameters is not stable.