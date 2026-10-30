"""Model-agnostic Real/Fake alignment by selecting batch membership."""

from .core import BatchAligner, AlignmentResult, shuffled_plan, balanced_plan
from .sampler import PlanBatchSampler

__version__ = "0.1.0"
__all__ = [
    "BatchAligner",
    "AlignmentResult",
    "PlanBatchSampler",
    "shuffled_plan",
    "balanced_plan",
]
