"""Equal-weight selection; retain every occurrence and every batch class slot.

This supports the authors' unbalanced shuffled batches as well as balanced
batches. A batch lacking either class is left untouched (distance undefined).
No validation/test attributes, labels or predictions enter this module.
"""

import numpy as np

ATTRIBUTES = [
    "face_area",
    "face_y",
    "abs_yaw",
    "log2_pixel_area",
    "sharpness",
    "border_black_fraction",
]


def histogram_state(plan, categories, labels, cardinalities):
    valid = plan >= 0
    y = np.where(valid, labels[np.maximum(plan, 0)], -1)
    counts = np.stack([(y == c).sum(1) for c in (0, 1)], axis=1)
    eligible = (counts > 0).all(1)
    delta = np.zeros((len(plan), categories.shape[1], max(cardinalities)), np.float64)
    a = np.arange(categories.shape[1])
    for b in np.flatnonzero(eligible):
        for image in plan[b, valid[b]]:
            c = int(labels[image])
            delta[b, a, categories[image]] += (1 if c == 0 else -1) / counts[b, c]
    return delta, counts, eligible, y


def distances(plan, categories, labels, cardinalities):
    delta, _, eligible, _ = histogram_state(plan, categories, labels, cardinalities)
    result = np.linalg.norm(delta, axis=2)
    result[~eligible] = np.nan
    return result


def align_pool(
    plan,
    categories,
    labels,
    cardinalities,
    scale,
    rng,
    rounds=64,
    proposals=128,
    weights=None,
):
    result = plan.copy()
    h, counts, eligible, y = histogram_state(result, categories, labels, cardinalities)
    squared = (h * h).sum(2)
    a = np.arange(categories.shape[1])
    width = plan.shape[1]
    positions = [np.flatnonzero(((y == c) & eligible[:, None]).ravel()) for c in (0, 1)]
    coefficient = (
        np.full(categories.shape[1], 1 / categories.shape[1])
        if weights is None
        else np.asarray(weights)
    ) / scale
    flat = result.ravel()
    accepted = 0
    for _ in range(rounds):
        c = int(rng.integers(2))
        if len(positions[c]) < 2:
            continue
        first = rng.choice(positions[c], proposals)
        second = rng.choice(positions[c], proposals)
        b1, b2 = first // width, second // width
        c1, c2 = categories[flat[first]], categories[flat[second]]
        t1 = ((1 if c == 0 else -1) / counts[b1, c])[:, None]
        t2 = ((1 if c == 0 else -1) / counts[b2, c])[:, None]
        changed = c1 != c2
        d1 = np.where(
            changed,
            2 * t1 * (h[b1[:, None], a, c2] - h[b1[:, None], a, c1]) + 2 * t1 * t1,
            0,
        )
        d2 = np.where(
            changed,
            2 * t2 * (h[b2[:, None], a, c1] - h[b2[:, None], a, c2]) + 2 * t2 * t2,
            0,
        )
        new1, new2 = np.maximum(squared[b1] + d1, 0), np.maximum(squared[b2] + d2, 0)
        change = (
            (
                np.sqrt(new1)
                + np.sqrt(new2)
                - np.sqrt(squared[b1])
                - np.sqrt(squared[b2])
            )
            * coefficient
        ).sum(1)
        change[b1 == b2] = np.inf
        best = int(change.argmin())
        if change[best] >= -1e-12:
            continue
        left, right = int(b1[best]), int(b2[best])
        old, new = c1[best], c2[best]
        h[left, a, old] -= t1[best, 0]
        h[left, a, new] += t1[best, 0]
        h[right, a, new] -= t2[best, 0]
        h[right, a, old] += t2[best, 0]
        squared[left], squared[right] = new1[best], new2[best]
        i, j = int(first[best]), int(second[best])
        flat[i], flat[j] = int(flat[j]), int(flat[i])
        accepted += 1
    return result, accepted


def make_plan(
    original,
    categories,
    labels,
    cardinalities,
    scale,
    epoch,
    pool_batches=8,
    rounds=64,
    proposals=128,
    seed=88111,
    weights=None,
):
    aligned = original.copy()
    swaps = 0
    for pool_id, start in enumerate(range(0, len(original), pool_batches)):
        aligned[start : start + pool_batches], count = align_pool(
            original[start : start + pool_batches],
            categories,
            labels,
            cardinalities,
            scale,
            np.random.default_rng(seed + epoch * 10000 + pool_id),
            rounds=rounds,
            proposals=proposals,
            weights=weights,
        )
        swaps += count
    # Preserve duplicate image occurrences introduced by cycling samplers.
    assert np.array_equal(np.sort(original.ravel()), np.sort(aligned.ravel()))
    before_y = np.where(original >= 0, labels[np.maximum(original, 0)], -1)
    after_y = np.where(aligned >= 0, labels[np.maximum(aligned, 0)], -1)
    assert np.array_equal(before_y, after_y)
    before = distances(original, categories, labels, cardinalities)
    after = distances(aligned, categories, labels, cardinalities)
    for start in range(0, len(original), pool_batches):
        assert (
            np.nansum(
                after[start : start + pool_batches]
                / scale
                * (1 if weights is None else weights)
            )
            <= np.nansum(
                before[start : start + pool_batches]
                / scale
                * (1 if weights is None else weights)
            )
            + 1e-8
        )
    return aligned, before, after, swaps


def bin_values(values, edges):
    output = np.empty(values.shape, np.int16)
    for a, edge in enumerate(edges):
        output[:, a] = np.clip(
            np.searchsorted(edge, values[:, a], side="right") - 1, 0, len(edge) - 2
        )
        output[~np.isfinite(values[:, a]), a] = len(edge) - 1
    return output
