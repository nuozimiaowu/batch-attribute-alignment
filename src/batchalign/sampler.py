"""A fixed-plan batch sampler accepted by PyTorch DataLoader (no torch import)."""

import numpy as np


class PlanBatchSampler:
    """Yield fixed dataset row indices; -1 padding is omitted, duplicates retained."""

    def __init__(self, plan):
        raw = np.asarray(plan)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("plan must be a two-dimensional integer array")
        if np.any(raw < -1) or np.any((raw >= 0).sum(axis=1) == 0):
            raise ValueError("only -1 padding is allowed, and batches cannot be empty")
        self.plan = raw.astype(np.int64, copy=True)
        self.plan.setflags(write=False)

    def __iter__(self):
        for row in self.plan:
            yield row[row >= 0].tolist()

    def __len__(self):
        return len(self.plan)
