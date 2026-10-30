"""CPU DataLoader integration. Replace ToyDataset with your existing dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from batchalign import BatchAligner, PlanBatchSampler, balanced_plan
from batchalign.torch_utils import image_rng


class ToyDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.epoch = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with image_rng(index, self.epoch, base_seed=7):
            image = torch.rand(3, 8, 8)  # Stand-in for a per-image CPU transform.
        return image, int(self.labels[index])


def main():
    labels = np.repeat([0, 1], 64)
    attributes = np.random.default_rng(0).normal(size=(len(labels), 6))
    reference = balanced_plan(labels, 16, seed=0)
    aligner = BatchAligner().fit(attributes, labels, reference)
    dataset = ToyDataset(labels)
    for epoch in range(2):
        baseline = balanced_plan(labels, 16, seed=epoch)
        result = aligner.align(baseline, epoch=epoch)
        dataset.epoch = epoch
        # Recreate workers each epoch; persistent workers need explicit epoch sharing.
        loader = DataLoader(
            dataset, batch_sampler=PlanBatchSampler(result.aligned_plan), num_workers=0
        )
        count = sum(len(y) for _, y in loader)
        print(f"epoch={epoch} occurrences={count} swaps={result.accepted_swaps}")


if __name__ == "__main__":
    main()
