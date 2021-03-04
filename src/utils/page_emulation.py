import math
import numpy as np


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    r, h = array.shape
    if r % nrows != 0:
        padding = (math.ceil(r / nrows) * nrows) - r
        array = np.vstack((array, np.zeros((padding, h))))
        r, h = array.shape
    if h % ncols != 0:
        padding = (math.ceil(h / ncols) * ncols) - h
        array = np.hstack((array, np.zeros((r, padding))))
        r, h = array.shape
    num_x_blocks = math.ceil(r / float(nrows))
    num_y_blocks = math.ceil(h / float(ncols))

    rows = np.vsplit(array, num_x_blocks)
    return [np.array(np.hsplit(row, num_y_blocks)) for row in rows]

# Reverts gather_blocks_to_pages
def pages_to_blocks(pages):
    blocks = []
    for p in pages:
        blocks.extend(p)
    return blocks


def gather_blocks_to_pages(splits, num_elem_per_page):
    blocks = np.concatenate(splits)
    pages = []
    i = 0

    while i < len(blocks):
        count = 0
        page = []
        while i < len(blocks) and \
            count + blocks[i].shape[0] * blocks[i].shape[1] <= num_elem_per_page:
            page.append(blocks[i])
            count += blocks[i].shape[0] * blocks[i].shape[1]
            i += 1
        pages.append(page)
        print("Adding {} elems to page {}".format(count, len(pages) - 1))
    return pages


# We assume that if one page has lesser blocks than the other,
# We should see if the smaller one matches the bigger one from
# the start, and not anywhere in between
def page_similarity(ps1, ps2):
    sim = np.zeros((len(ps1), len(ps2)))

    for i, p1 in enumerate(ps1):
        for j, p2 in enumerate(ps2):
            k = min(len(p1), len(p2))
            a = np.array(p1[:k])
            b = np.array(p2[:k])
            c = np.count_nonzero(np.absolute(a - b) <= 0.01)
            sim[i][j] = c / a.size

    return sim


def merge_blocks(blocks, num_blocks_x, num_blocks_y, x, y):
    b_x, b_y = blocks[0].shape
    t_x, t_y = (
        b_x * num_blocks_x,
        b_y * num_blocks_y,
    )
    rows = [
        np.hstack(blocks[i * num_blocks_y:i * num_blocks_y + num_blocks_y])
        for i in range(num_blocks_x)
    ]
    matrix = np.vstack(rows)
    assert matrix.shape[0] == t_x
    r_x = t_x - x
    r_y = t_y - y
    if r_x == 0 and r_y == 0:
        return matrix
    elif r_x == 0:
        return matrix[:, :-r_y]
    elif r_y == 0:
        return matrix[:-r_x, :]
    else:
        return matrix[:-r_x, :-r_y]


def merge_pages(p1, p2):
    ps = []
    for i, p in enumerate(p1):
        if i >= len(p2):
            ps.append(np.array(p))
        else:
            ps.append((p + p2[i]) / 2)
    return ps


def combine_similar_pages(ps1, ps2, sim_scores, threshold=0.9):
    new_ps1 = [None] * len(ps1)
    new_ps2 = [None] * len(ps2)

    for ps1_idx, scores in enumerate(sim_scores):
        if np.max(scores) >= threshold:
            ps2_idx = np.argmax(scores)
            print("PS1: Merging {} and {}".format(ps1_idx, ps2_idx))
            new_ps1[ps1_idx] = merge_pages(ps1[ps1_idx], ps2[ps2_idx])
            new_ps2[ps2_idx] = ps1_idx
        else:
            # No need to make new copies here since we should not be using ps1 or ps2 anymore
            new_ps1[ps1_idx] = ps1[ps1_idx]

    for ps2_idx in range(len(ps2)):
        if new_ps2[ps2_idx] is not None:
            ps1_idx = new_ps2[ps2_idx]
            print("PS2: Merging {} and {}".format(ps2_idx, ps1_idx))
            new_ps2[ps2_idx] = merge_pages(ps2[ps2_idx], ps1[ps1_idx])
        else:
            # No need to make new copies here since we should not be using ps1 or ps2 anymore
            new_ps2[ps2_idx] = ps2[ps2_idx]

    return new_ps1, new_ps2


def weight_sim(w1, w2):
    nw1 = w1.flatten()
    nw2 = w2.flatten()
    if nw1.size < nw2.size:
        rem = nw2.size - nw1.size
        nw2 = nw2[:-rem]
    elif nw1.size > nw2.size:
        rem = nw1.size - nw2.size
        nw1 = nw1[:-rem]

    return np.count_nonzero(np.absolute(nw2 - nw1) <= 0.01) / nw1.size


def share_weights(w1, w2, a, b, c, t):
    m, n = w1.shape
    x = split(w1, a, b)
    bx, by = len(x), x[0].shape[0]

    o, p = w2.shape
    y = split(w2, a, b)
    cx, cy = len(y), y[0].shape[0]

    ps1 = gather_blocks_to_pages(x, c)  # ~1MB 16 bytes * 63725
    ps2 = gather_blocks_to_pages(y, c)  # ~1MB 16 bytes * 63725

    sim = page_similarity(ps1, ps2)
    print(sim)

    nps1, nps2 = combine_similar_pages(ps1, ps2, sim, t)

    wb1 = pages_to_blocks(nps1)
    wb1 = merge_blocks(wb1, bx, by, m, n)

    wb2 = pages_to_blocks(nps2)
    wb2 = merge_blocks(wb2, cx, cy, o, p)

    return wb1, wb2