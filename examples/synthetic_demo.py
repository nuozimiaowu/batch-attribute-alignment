"""CPU-only method demonstration; these numbers are not detector results."""

import numpy as np

from batchalign import BatchAligner, balanced_plan


def main():
    rng = np.random.default_rng(12)
    labels = np.repeat([0, 1], 256)
    attributes = rng.normal(size=(len(labels), 6)) + labels[:, None] * 0.35
    plan = balanced_plan(labels, batch_size=32, seed=0)
    aligner = BatchAligner().fit(attributes, labels, plan)
    result = aligner.align(plan)
    assert np.array_equal(np.sort(plan.ravel()), np.sort(result.aligned_plan.ravel()))
    assert np.array_equal(labels[plan], labels[result.aligned_plan])
    summary = result.summary()
    print(
        f"Synthetic joint objective: {summary['cumulative_objective_before']:.4f} -> "
        f"{summary['cumulative_objective_after']:.4f}"
    )
    print(f"Accepted same-class swaps: {result.accepted_swaps}")


if __name__ == "__main__":
    main()
