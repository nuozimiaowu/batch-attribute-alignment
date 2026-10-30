"""Fit source-only histogram rules and align an existing epoch's batch plan."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from . import _kernel

STANDARD_ATTRIBUTES = (
    "face_area",
    "face_y",
    "abs_yaw",
    "log2_pixel_area",
    "sharpness",
    "border_black_fraction",
)


def _labels(labels):
    raw = np.asarray(labels)
    if raw.ndim != 1 or not len(raw) or not np.isin(raw, [0, 1]).all():
        raise ValueError("labels must be a nonempty 1-D binary array: 0=Real, 1=Fake")
    return raw.astype(np.int64, copy=True)


def _values(values, n):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n or array.shape[1] == 0:
        raise ValueError(
            "attributes must have shape [number of training images, number of attributes]"
        )
    return array.copy()


def _plan(plan, n):
    array = np.asarray(plan)
    if array.ndim != 2 or not array.size or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            "plan must be a nonempty 2-D integer array of dataset row indices"
        )
    if np.any(array < -1) or np.any(array >= n):
        raise ValueError("plan indices must be in [0,N); only -1 is allowed as padding")
    if np.any((array >= 0).sum(1) == 0):
        raise ValueError("empty padded batches are not supported")
    return array.astype(np.int64, copy=True)


def shuffled_plan(labels, batch_size=32, seed=0, drop_last=True):
    """Ordinary shuffled baseline. Existing native samplers can supply their own plan."""
    y = _labels(labels)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    indices = np.random.default_rng(seed).permutation(len(y))
    if drop_last:
        indices = indices[: len(indices) // batch_size * batch_size]
    elif len(indices) % batch_size:
        indices = np.pad(indices, (0, (-len(indices)) % batch_size), constant_values=-1)
    if not len(indices):
        raise ValueError(
            "no batches remain; decrease batch_size or set drop_last=False"
        )
    return indices.reshape(-1, batch_size)


def balanced_plan(labels, batch_size=32, seed=0, drop_last=True):
    """Example 1:1 baseline without replacement; oversampling is never implicit.

    Uses equal class totals limited by the smaller class. The selected occurrence
    multiset, including any omitted samples, is then identical in both arms.
    """
    y = _labels(labels)
    if not isinstance(batch_size, int) or batch_size < 2 or batch_size % 2:
        raise ValueError("balanced batch_size must be a positive even integer")
    rng = np.random.default_rng(seed)
    real, fake = [rng.permutation(np.flatnonzero(y == c)) for c in (0, 1)]
    n = min(len(real), len(fake))
    half = batch_size // 2
    if drop_last:
        n = n // half * half
    if n == 0:
        raise ValueError(
            "not enough samples of both classes for the requested balanced plan"
        )
    batches = []
    for start in range(0, n, half):
        row = rng.permutation(
            np.r_[
                real[start : min(start + half, n)], fake[start : min(start + half, n)]
            ]
        )
        batches.append(np.pad(row, (0, batch_size - len(row)), constant_values=-1))
    return np.stack(batches)


def _capped_weights(scores, cap):
    if not np.isfinite(cap) or cap <= 0 or cap > 1 or cap * len(scores) < 1 - 1e-12:
        raise ValueError(
            "cap must be in (0,1] and cap * number_of_attributes must be >= 1"
        )
    out = np.zeros(len(scores))
    available = np.ones(len(scores), bool)
    remaining = 1.0
    while available.any():
        positions = np.flatnonzero(available)
        s = scores[positions]
        proposal = remaining * (s / s.sum() if s.sum() else np.full(len(s), 1 / len(s)))
        exceeds = proposal > cap + 1e-14
        if not exceeds.any():
            out[positions] = proposal
            break
        fixed = positions[exceeds]
        out[fixed] = cap
        available[fixed] = False
        remaining -= cap * len(fixed)
    return out


def _edges(values, names, n_bins, fixed_edges):
    output = []
    for a, name in enumerate(names):
        v = values[:, a]
        finite = v[np.isfinite(v)]
        if fixed_edges and name in fixed_edges:
            edge = np.asarray(fixed_edges[name], dtype=np.float64)
        elif name in {"sharpness", "sharpness256_log1p"}:
            edge = np.linspace(0, 14, n_bins + 1)
        elif len(finite):
            edge = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
            if len(edge) == 1:
                edge = np.array([edge[0] - 0.5, edge[0] + 0.5])
            if name == "border_black_fraction" and np.any(finite == 0):
                edge = np.unique(np.r_[edge, np.nextafter(0.0, 1.0)])
        else:
            edge = np.array([-0.5, 0.5])
        if (
            edge.ndim != 1
            or len(edge) < 2
            or not np.isfinite(edge).all()
            or not (np.diff(edge) > 0).all()
        ):
            raise ValueError(f"invalid strictly increasing finite bin edges for {name}")
        output.append(edge.copy())
    return output


@dataclass
class AlignmentResult:
    original_plan: np.ndarray
    aligned_plan: np.ndarray
    before_l2: np.ndarray
    after_l2: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    attribute_names: tuple
    accepted_swaps: int
    epoch: int

    @property
    def valid_batches(self):
        return np.isfinite(self.before_l2).all(axis=1)

    @property
    def objective_before(self):
        return np.sum(self.before_l2 / self.scale * self.weights, axis=1)

    @property
    def objective_after(self):
        return np.sum(self.after_l2 / self.scale * self.weights, axis=1)

    def summary(self):
        before = np.nansum(self.before_l2, axis=0)
        after = np.nansum(self.after_l2, axis=0)
        left, right = float(np.nansum(self.objective_before)), float(
            np.nansum(self.objective_after)
        )
        return dict(
            epoch=self.epoch,
            batches=len(self.original_plan),
            valid_batches=int(self.valid_batches.sum()),
            undefined_batches=int((~self.valid_batches).sum()),
            accepted_swaps=self.accepted_swaps,
            attribute_names=list(self.attribute_names),
            weights=self.weights.tolist(),
            scale=self.scale.tolist(),
            cumulative_l2_before=before.tolist(),
            cumulative_l2_after=after.tolist(),
            cumulative_objective_before=left,
            cumulative_objective_after=right,
            objective_reduction_pct=100 * (1 - right / left) if left else None,
            per_attribute_reduction_pct=[
                100 * (1 - b / a) if a else None for a, b in zip(before, after)
            ],
        )

    def save(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "plans.npz",
            original=self.original_plan,
            aligned=self.aligned_plan,
            before_l2=self.before_l2,
            after_l2=self.after_l2,
            weights=self.weights,
            scale=self.scale,
        )
        (destination / "summary.json").write_text(
            json.dumps(self.summary(), indent=2, allow_nan=False) + "\n"
        )


class BatchAligner:
    """Fit training-only rules once, then change only batch membership.

    ``equal`` is the default. ``source_difference`` and its capped variant are
    experimental ablations, not claims of improved detection performance.
    All RNG streams are local; the detector's training RNG is untouched.
    """

    def __init__(
        self,
        strategy="equal",
        weights=None,
        n_bins=40,
        pool_batches=8,
        rounds=64,
        proposals=128,
        seed=88111,
        cap=1 / 3,
    ):
        if strategy not in {"equal", "source_difference", "capped_source_difference"}:
            raise ValueError("unknown weighting strategy")
        for name, value, minimum in [
            ("n_bins", n_bins, 1),
            ("pool_batches", pool_batches, 1),
            ("rounds", rounds, 0),
            ("proposals", proposals, 1),
        ]:
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid {name}")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.strategy, self.manual_weights = strategy, weights
        self.n_bins, self.pool_batches, self.rounds, self.proposals = (
            n_bins,
            pool_batches,
            rounds,
            proposals,
        )
        self.seed, self.cap = seed, cap

    def fit(
        self, attributes, labels, reference_plan, attribute_names=None, fixed_edges=None
    ):
        """Freeze bins, weights and scales from source-training data only."""
        self.labels_ = _labels(labels)
        values = _values(attributes, len(self.labels_))
        count = values.shape[1]
        self.attribute_names_ = (
            tuple(attribute_names)
            if attribute_names is not None
            else tuple(f"attribute_{a}" for a in range(count))
        )
        if (
            len(self.attribute_names_) != count
            or len(set(self.attribute_names_)) != count
        ):
            raise ValueError(
                "attribute_names must be unique and match the number of columns"
            )
        plan = _plan(reference_plan, len(self.labels_))
        self.edges_ = _edges(values, self.attribute_names_, self.n_bins, fixed_edges)
        categories = _kernel.bin_values(values, self.edges_)
        cardinalities = [len(e) for e in self.edges_]
        distance = _kernel.distances(plan, categories, self.labels_, cardinalities)
        if not np.isfinite(distance).all(axis=1).any():
            raise ValueError(
                "reference plan has no batch containing both Real and Fake"
            )
        self.scale_ = np.maximum(np.nanmean(distance, axis=0), 1e-12)
        selected = plan[plan >= 0]
        real, fake = (
            selected[self.labels_[selected] == 0],
            selected[self.labels_[selected] == 1],
        )
        difference = []
        for a, n in enumerate(cardinalities):
            p = np.bincount(categories[real, a], minlength=n) / len(real)
            q = np.bincount(categories[fake, a], minlength=n) / len(fake)
            difference.append(np.linalg.norm(p - q))
        self.source_difference_ = np.array(difference)
        score = self.source_difference_ / self.scale_
        if self.manual_weights is not None:
            weight = np.asarray(self.manual_weights, dtype=np.float64)
            if (
                weight.shape != (count,)
                or not np.isfinite(weight).all()
                or np.any(weight < 0)
                or weight.sum() <= 0
            ):
                raise ValueError(
                    "weights must be finite, nonnegative, nonzero and match attribute columns"
                )
            self.weights_ = weight / weight.sum()
        elif self.strategy == "equal":
            self.weights_ = np.full(count, 1 / count)
        elif self.strategy == "source_difference":
            self.weights_ = (
                score / score.sum() if score.sum() else np.full(count, 1 / count)
            )
        else:
            self.weights_ = _capped_weights(score, self.cap)
        self.training_values_ = values
        self.fit_fingerprint_ = hashlib.sha256(
            values.tobytes() + self.labels_.tobytes() + plan.tobytes()
        ).hexdigest()
        return self

    def align(self, original_plan, attributes=None, epoch=0):
        if not hasattr(self, "labels_"):
            raise RuntimeError("call fit on source-training data first")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        plan = _plan(original_plan, len(self.labels_))
        if (len(plan) + self.pool_batches - 1) // self.pool_batches > 10000:
            raise ValueError(
                "more than 10,000 pools would overlap the historical epoch RNG stride; increase pool_batches"
            )
        values = (
            self.training_values_
            if attributes is None
            else _values(attributes, len(self.labels_))
        )
        if values is None:
            raise ValueError("loaded aligners require attributes for each align call")
        if values.shape[1] != len(self.attribute_names_):
            raise ValueError("attribute column count changed")
        categories = _kernel.bin_values(values, self.edges_)
        aligned, before, after, swaps = _kernel.make_plan(
            plan,
            categories,
            self.labels_,
            [len(e) for e in self.edges_],
            self.scale_,
            epoch,
            pool_batches=self.pool_batches,
            rounds=self.rounds,
            proposals=self.proposals,
            seed=self.seed,
            weights=self.weights_,
        )
        return AlignmentResult(
            plan,
            aligned,
            before,
            after,
            self.scale_.copy(),
            self.weights_.copy(),
            self.attribute_names_,
            swaps,
            epoch,
        )

    def save(self, path):
        if not hasattr(self, "labels_"):
            raise RuntimeError("fit before saving")
        metadata = dict(
            strategy=self.strategy,
            n_bins=self.n_bins,
            pool_batches=self.pool_batches,
            rounds=self.rounds,
            proposals=self.proposals,
            seed=self.seed,
            cap=self.cap,
            names=self.attribute_names_,
            fingerprint=self.fit_fingerprint_,
        )
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.array(json.dumps(metadata)),
                labels=self.labels_,
                scale=self.scale_,
                weights=self.weights_,
                difference=self.source_difference_,
                **{f"edge_{a}": e for a, e in enumerate(self.edges_)},
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            obj = cls(
                **{
                    k: meta[k]
                    for k in [
                        "strategy",
                        "n_bins",
                        "pool_batches",
                        "rounds",
                        "proposals",
                        "seed",
                        "cap",
                    ]
                }
            )
            obj.attribute_names_ = tuple(meta["names"])
            obj.fit_fingerprint_ = meta["fingerprint"]
            obj.labels_ = data["labels"].copy()
            obj.scale_ = data["scale"].copy()
            obj.weights_ = data["weights"].copy()
            obj.source_difference_ = data["difference"].copy()
            obj.edges_ = [
                data[f"edge_{a}"].copy() for a in range(len(obj.attribute_names_))
            ]
            obj.training_values_ = None
        return obj
