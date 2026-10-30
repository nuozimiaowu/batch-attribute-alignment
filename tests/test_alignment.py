import json
import random

import numpy as np
import pytest

from batchalign import BatchAligner, PlanBatchSampler, balanced_plan, shuffled_plan


def fixture():
    y = np.repeat([0, 1], 64)
    x = np.random.default_rng(4).normal(size=(128, 6)) + y[:, None] * 0.4
    return x, y, balanced_plan(y, 16, seed=8)


def independent_distances(plan, x, y, edges):
    answer = np.full((len(plan), x.shape[1]), np.nan)
    for b, row in enumerate(plan):
        row = row[row >= 0]
        if set(y[row]) != {0, 1}:
            continue
        for a, edge in enumerate(edges):
            hist = []
            for c in (0, 1):
                v = x[row[y[row] == c], a]
                bins = np.searchsorted(edge, v, side="right") - 1
                bins = np.clip(bins, 0, len(edge) - 2)
                bins[~np.isfinite(v)] = len(edge) - 1
                hist.append(np.bincount(bins, minlength=len(edge)) / len(v))
            answer[b, a] = np.sqrt(np.sum((hist[0] - hist[1]) ** 2))
    return answer


def test_objective_and_independent_histograms():
    x, y, plan = fixture()
    aligner = BatchAligner(pool_batches=4).fit(x, y, plan)
    result = aligner.align(plan)
    assert result.accepted_swaps > 0
    assert np.array_equal(np.sort(plan.ravel()), np.sort(result.aligned_plan.ravel()))
    assert np.array_equal(y[plan], y[result.aligned_plan])
    assert np.array_equal(result.aligned_plan, aligner.align(plan).aligned_plan)
    np.testing.assert_allclose(
        result.before_l2, independent_distances(plan, x, y, aligner.edges_), atol=1e-14
    )
    np.testing.assert_allclose(
        result.after_l2,
        independent_distances(result.aligned_plan, x, y, aligner.edges_),
        atol=1e-14,
    )
    for start in range(0, len(plan), 4):
        assert (
            result.objective_after[start : start + 4].sum()
            <= result.objective_before[start : start + 4].sum() + 1e-10
        )


def test_unequal_counts_duplicates_padding_single_class():
    x, y, _ = fixture()
    plan = np.array([[0, 0, 64, -1], [1, 65, 66, -1], [2, 3, -1, -1], [67, 68, 69, -1]])
    aligner = BatchAligner().fit(x, y, plan)
    result = aligner.align(plan)
    np.testing.assert_array_equal(
        np.sort(plan.ravel()), np.sort(result.aligned_plan.ravel())
    )
    np.testing.assert_array_equal(plan == -1, result.aligned_plan == -1)
    np.testing.assert_array_equal(plan[2:], result.aligned_plan[2:])
    np.testing.assert_array_equal(
        np.where(plan >= 0, y[np.maximum(plan, 0)], -1),
        np.where(plan >= 0, y[np.maximum(result.aligned_plan, 0)], -1),
    )
    assert result.summary()["undefined_batches"] == 2
    assert np.isnan(result.after_l2[2:]).all()


def test_missing_constant_and_single_attribute():
    x, y, plan = fixture()
    x[:, 0] = 1
    x[:, 1] = np.nan
    x[::3, 2] = np.inf
    fit = BatchAligner().fit(x, y, plan)
    result = fit.align(plan)
    assert np.isfinite(result.after_l2).all()
    assert np.all(result.after_l2[:, :2] == 0)
    assert BatchAligner().fit(x[:, 0], y, plan).align(plan).after_l2.shape == (
        len(plan),
        1,
    )


def test_fixed_edges_and_sharpness():
    x, y, plan = fixture()
    fit = BatchAligner().fit(x[:, :1], y, plan, ["sharpness"])
    np.testing.assert_array_equal(fit.edges_[0], np.linspace(0, 14, 41))
    fit = BatchAligner().fit(x[:, :1], y, plan, ["custom"], {"custom": [0, 1, 2]})
    np.testing.assert_array_equal(fit.edges_[0], [0, 1, 2])


@pytest.mark.parametrize(
    "strategy", ["equal", "source_difference", "capped_source_difference"]
)
def test_serialization_and_strategies(tmp_path, strategy):
    x, y, plan = fixture()
    fit = BatchAligner(strategy=strategy).fit(x, y, plan)
    assert fit.weights_.sum() == pytest.approx(1)
    if strategy.startswith("capped"):
        assert fit.weights_.max() <= 1 / 3 + 1e-12
    fit.save(tmp_path / "fit.npz")
    loaded = BatchAligner.load(tmp_path / "fit.npz")
    np.testing.assert_array_equal(
        fit.align(plan).aligned_plan, loaded.align(plan, x).aligned_plan
    )
    loaded.align(plan, x).save(tmp_path / "result")
    assert json.loads((tmp_path / "result/summary.json").read_text())["batches"] == len(
        plan
    )
    with pytest.raises(ValueError):
        loaded.align(plan)


def test_baseline_helpers_and_no_rng_side_effects():
    x, y, plan = fixture()
    np.random.seed(77)
    random.seed(77)
    expected_numpy = np.random.get_state()
    expected_python = random.getstate()
    BatchAligner().fit(x, y, plan).align(plan)
    state = np.random.get_state()
    assert state[0] == expected_numpy[0]
    np.testing.assert_array_equal(state[1], expected_numpy[1])
    assert state[2:] == expected_numpy[2:]
    assert random.getstate() == expected_python
    p = balanced_plan(y, 30, drop_last=False)
    assert np.all(np.sum(y[np.maximum(p, 0)] * (p >= 0), axis=1) == (p >= 0).sum(1) / 2)
    p = shuffled_plan(y, 30, drop_last=False)
    np.testing.assert_array_equal(np.sort(p[p >= 0]), np.arange(len(y)))
    assert list(PlanBatchSampler(np.array([[1, 1, -1], [2, -1, -1]]))) == [[1, 1], [2]]


def test_round_zero_and_weights():
    x, y, plan = fixture()
    fit = BatchAligner(rounds=0, weights=[1, 0, 0, 0, 0, 0]).fit(x, y, plan)
    np.testing.assert_array_equal(fit.align(plan).aligned_plan, plan)
    assert fit.weights_[0] == 1


@pytest.mark.parametrize(
    "bad", [np.array([[128]]), np.array([[-2]]), np.array([[-1]]), np.array([[0.5]])]
)
def test_reject_bad_plan(bad):
    x, y, _ = fixture()
    with pytest.raises(ValueError):
        BatchAligner().fit(x, y, bad)


def test_reject_bad_configuration():
    x, y, plan = fixture()
    with pytest.raises(ValueError):
        BatchAligner(weights=[0] * 6).fit(x, y, plan)
    with pytest.raises(ValueError):
        BatchAligner(strategy="capped_source_difference", cap=0.1).fit(x, y, plan)
    with pytest.raises(ValueError):
        BatchAligner().fit(x, y, plan, ["repeated"] * 6)
    with pytest.raises(ValueError):
        BatchAligner().fit(x, y, np.array([[0, 1], [64, 65]]))
    with pytest.raises(ValueError):
        BatchAligner().fit(x, y + 1, plan)
    with pytest.raises(RuntimeError):
        BatchAligner().align(plan)


def test_cli(tmp_path):
    from batchalign.cli import main

    csv = tmp_path / "training.csv"
    csv.write_text(
        "id,label,attribute\n"
        + "".join(f"sample{i},{i%2},{i/100}\n" for i in range(64))
    )
    out = tmp_path / "output"
    main(
        [
            str(csv),
            "--attributes",
            "attribute",
            "--balanced",
            "--batch-size",
            "8",
            "--output",
            str(out),
        ]
    )
    assert (out / "plans.npz").exists()
    assert (out / "row_mapping.csv").read_text().splitlines()[1] == "0,sample0,0"


def test_optional_torch_rng():
    torch = pytest.importorskip("torch")
    from batchalign.torch_utils import image_rng

    before = torch.random.get_rng_state().clone()
    with image_rng(5, 2):
        a = torch.rand(4)
    with image_rng(5, 2):
        b = torch.rand(4)
    assert torch.equal(a, b)
    assert torch.equal(before, torch.random.get_rng_state())
