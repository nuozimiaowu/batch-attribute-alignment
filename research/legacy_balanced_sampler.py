"""Source-only weights and exposure-preserving multi-attribute batch selection.

No detector scores, validation samples or test data enter this module.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ATTRIBUTES = ['face_area', 'face_y', 'abs_yaw', 'log2_pixel_area', 'sharpness256_log1p', 'border_black_fraction']
ATTRIBUTE_KEYS = ['face_size', 'face_position', 'head_pose', 'resolution', 'sharpness', 'border']
ARMS = ['equal', 'source_difference', 'capped_source_difference']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def bin_matrix(values, edges):
    """One categorical value per image/attribute; missing gets its own bin."""
    result = np.empty(values.shape, np.int16)
    for a, boundary in enumerate(edges):
        v = values[:, a]
        result[:, a] = np.clip(np.searchsorted(boundary, v, side='right') - 1, 0, len(boundary) - 2)
        result[~np.isfinite(v), a] = len(boundary) - 1
    return result


def batch_distances(plan, categories, labels, cardinalities):
    """Return [batch, attribute] separately normalized histogram L2 distances."""
    plan = np.asarray(plan, np.int64)
    y = labels[plan]
    assert np.all((y == 0).sum(1) == 16) and np.all((y == 1).sum(1) == 16)
    result = np.empty((len(plan), categories.shape[1]), np.float64)
    for a, count in enumerate(cardinalities):
        delta = np.zeros((len(plan), count), np.float64)
        np.add.at(delta, (np.repeat(np.arange(len(plan)), 32), categories[plan, a].ravel()),
                  np.where(y.ravel() == 0, 1/16, -1/16))
        result[:, a] = np.linalg.norm(delta, axis=1)
    return result


def capped_weights(scores, cap=1/3):
    """Proportional allocation on a capped simplex, including zero-score cases."""
    scores = np.asarray(scores, np.float64)
    assert np.isfinite(scores).all() and (scores >= 0).all()
    assert cap * len(scores) >= 1 - 1e-12
    output = np.zeros(len(scores), np.float64)
    available = np.ones(len(scores), bool)
    remaining = 1.
    while available.any():
        positions = np.flatnonzero(available)
        s = scores[positions]
        proposal = remaining * (s / s.sum() if s.sum() else np.full(len(s), 1/len(s)))
        exceeds = proposal > cap + 1e-14
        if not exceeds.any():
            output[positions] = proposal
            break
        fixed = positions[exceeds]
        output[fixed] = cap
        available[fixed] = False
        remaining -= cap * len(fixed)
    assert abs(output.sum() - 1) < 1e-12 and output.max() <= cap + 1e-12
    return output


def source_weights(baseline, categories, labels, cardinalities):
    """Freeze all rules from exactly the same balanced epoch-0 training pool."""
    selected = baseline.ravel()
    assert len(np.unique(selected)) == len(selected)
    random_distance = batch_distances(baseline, categories, labels, cardinalities)
    scale = np.maximum(random_distance.mean(axis=0), 1e-12)
    discrepancy = []
    for a, count in enumerate(cardinalities):
        real = selected[labels[selected] == 0]; fake = selected[labels[selected] == 1]
        p = np.bincount(categories[real, a], minlength=count) / len(real)
        q = np.bincount(categories[fake, a], minlength=count) / len(fake)
        discrepancy.append(float(np.linalg.norm(p-q)))
    score = np.asarray(discrepancy) / scale
    weights = dict(equal=np.full(len(scale), 1/len(scale)),
        source_difference=score/score.sum() if score.sum() else np.full(len(scale), 1/len(scale)),
        capped_source_difference=capped_weights(score))
    return scale, np.asarray(discrepancy), score, weights


def optimize_pool(pool, categories, labels, cardinalities, weights, scale, rng,
                  rounds=64, proposals=128):
    """Greedy same-class swaps across <=8 batches, always reducing the objective.

    Candidate changes use exact squared-histogram-count updates, so a proposal
    costs O(attributes), not O(attributes * bins). At most one swap is accepted
    per round. Identical proposal streams are used for all weighting rules.
    """
    pool = pool.copy()
    if len(pool) < 2:
        return pool, 0
    y = labels[pool]
    signed = np.zeros((len(pool), categories.shape[1], max(cardinalities)), np.int32)
    a = np.arange(categories.shape[1])
    for b, row in enumerate(pool):
        for i, image in enumerate(row):
            signed[b, a, categories[image]] += 1 if y[b, i] == 0 else -1
    squared = np.square(signed).sum(axis=2)
    coefficient = weights / (16 * scale)
    positions = [np.flatnonzero(y.ravel() == label) for label in [0, 1]]
    flat = pool.ravel()
    accepted = 0
    for _ in range(rounds):
        label = int(rng.integers(2))
        sign = 1 if label == 0 else -1
        first = rng.choice(positions[label], proposals)
        second = rng.choice(positions[label], proposals)
        b1, b2 = first // 32, second // 32
        c1, c2 = categories[flat[first]], categories[flat[second]]
        different = c1 != c2
        d1 = np.where(different, 2*sign*(signed[b1[:, None], a, c2] - signed[b1[:, None], a, c1]) + 2, 0)
        d2 = np.where(different, 2*sign*(signed[b2[:, None], a, c1] - signed[b2[:, None], a, c2]) + 2, 0)
        new1 = squared[b1] + d1; new2 = squared[b2] + d2
        valid = b1 != b2
        assert (new1[valid] >= 0).all() and (new2[valid] >= 0).all()
        improvement = ((np.sqrt(np.maximum(new1, 0)) + np.sqrt(np.maximum(new2, 0))
                        - np.sqrt(squared[b1]) - np.sqrt(squared[b2])) * coefficient).sum(axis=1)
        improvement[~valid] = np.inf
        best = int(np.argmin(improvement))
        if improvement[best] >= -1e-12:
            continue
        left, right = int(b1[best]), int(b2[best])
        old, new = c1[best], c2[best]
        signed[left, a, old] -= sign; signed[left, a, new] += sign
        signed[right, a, new] -= sign; signed[right, a, old] += sign
        squared[left] = new1[best]; squared[right] = new2[best]
        i, j = int(first[best]), int(second[best])
        flat[i], flat[j] = int(flat[j]), int(flat[i])
        accepted += 1
    assert np.array_equal(labels[pool], y)
    return pool, accepted


def make_plan(baseline, categories, labels, cardinalities, weights, scale, epoch,
              pool_batches=8, rounds=64, proposals=128):
    result = np.empty_like(baseline)
    swaps = 0
    for pool_id, start in enumerate(range(0, len(baseline), pool_batches)):
        rng = np.random.default_rng(88111 + epoch * 10000 + pool_id)
        result[start:start+pool_batches], accepted = optimize_pool(
            baseline[start:start+pool_batches], categories, labels, cardinalities,
            weights, scale, rng, rounds=rounds, proposals=proposals)
        swaps += accepted
    assert np.array_equal(np.sort(result.ravel()), np.sort(baseline.ravel()))
    assert np.array_equal(labels[result], labels[baseline])
    before = batch_distances(baseline, categories, labels, cardinalities)
    after = batch_distances(result, categories, labels, cardinalities)
    objective_before = (before / scale * weights).sum(axis=1)
    objective_after = (after / scale * weights).sum(axis=1)
    for start in range(0, len(baseline), pool_batches):
        assert objective_after[start:start+pool_batches].sum() <= objective_before[start:start+pool_batches].sum() + 1e-9
    return result, after, objective_after, before, objective_before, swaps
