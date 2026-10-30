# Original balanced reference

`legacy_balanced_sampler.py` preserves the research implementation used for
balanced multi-attribute selection. It expects complete balanced batches and
uses integer histogram counts during swap search. It is not the public API.

The public NumPy kernel works directly with class-normalized histogram masses
and also accepts unequal class counts, repeated indices and padding. Floating
point ties can choose different equally scoring swaps; do not assume an
identical index sequence between these two numerical implementations.
Use the legacy module with the original epoch plans, bins, weights and scales
when auditing a historical balanced run. Both implement the same histogram
L2 objective and preserve class slots and epoch occurrences.
