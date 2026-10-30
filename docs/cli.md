# CSV interface

Example training-only CSV (each row is one dataset index):

```csv
id,label,face_area,sharpness
image_000,0,0.25,3.1
image_001,1,0.28,2.8
```

Supply enough rows for batches containing both classes. Blank attributes or
nonfinite numeric values use the missing bin. IDs must be unique. No image
files are read by this command.

```bash
batchalign training.csv --attributes face_area sharpness --balanced \
    --batch-size 32 --output outputs/epoch0
```

For an existing training sampler, save its baseline 2-D index array as a NumPy
file and pass `--plan baseline.npy` instead of `--balanced`. The default when
neither flag is supplied is ordinary unbalanced shuffling with drop-last.

For later epochs, reuse the fitted rules:

```bash
batchalign training.csv --attributes face_area sharpness --plan epoch1.npy \
    --fit-state outputs/epoch0/fit.npz --epoch 1 --output outputs/epoch1
```

Retain exactly the same CSV ID ordering, labels, columns and measurement
contract when reusing state. The API state contains labels but does not embed
your private image ID mapping; label equality alone cannot detect permutation
of same-class image rows. Archive `row_mapping.csv` with each experiment.
Outputs are never silently overwritten. The CLI writes the baseline and
aligned plans, per-batch distances, summary, fitted rules and invocation.
