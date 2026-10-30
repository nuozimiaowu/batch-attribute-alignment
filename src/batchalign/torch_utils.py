"""Optional per-image augmentation replay; import requires PyTorch."""

from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def image_rng(index, epoch, base_seed=0):
    """Isolate CPU transform RNG by stable image index and epoch.

    Use around CPU augmentation only. Restore the caller's Python, NumPy and
    CPU Torch RNG states, including when num_workers=0. This does not control
    transforms with private generators or CUDA-side augmentation.
    """
    seed = (int(base_seed) + int(index) * 97 + int(epoch) * 1000003) % 2**32
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed)
            # Seed CPU only: do not modify CUDA model RNG.
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
