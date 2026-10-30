# Release validation

Release preparation, 2026-10-30:

- Nova CPU: 16 tests passed, including the optional PyTorch RNG test.
- Windows CPU: 15 tests passed; optional PyTorch test skipped because Torch
  was not installed in that Python environment.
- Source and wheel distributions built successfully in an isolated build environment.
- Wheel installed into a separate target directory; the CPU DataLoader example
  completed two epochs with 128 occurrences each.
- Synthetic example joint objective: 16.0000 to 12.5520, 127 accepted swaps.

The tests independently reconstruct normalized Real/Fake histograms and verify
their L2 distances. They cover class-slot and occurrence preservation, duplicate
indices, padding, unequal class counts, single-class exclusions, missing and
constant attributes, fixed edges, deterministic local RNG, serialization,
weighting modes, invalid inputs and the CSV command.

These are software checks, not new model training or independent replication of
the recorded Xception metrics. No GPUs were allocated for release validation.
