# Method and controlled comparison

Let p(b,a,c,k) be the fraction of class c occurrences in batch b assigned to
bin k of attribute a. The discrepancy is

```
d(b,a) = sqrt(sum_k (p(b,a,Real,k) - p(b,a,Fake,k))**2)
s(a)   = max(mean_b d(reference_batch_b,a), 1e-12)
J(b)   = sum_a w(a) * d(b,a) / s(a)
```

This is a difference between two normalized histograms. It is neither an
individual class variance nor a raw-count distance. Missing/nonfinite values
have a dedicated bin. Batches lacking either class have undefined distance
(NaN), are excluded from cumulative sums, and remain unchanged. Their count
is explicitly reported.

Fit once using source training only. Default bins use 40 pooled quantiles,
with repeated edges deduplicated. Attribute `sharpness` (also accepted as
`sharpness256_log1p`) uses 40 fixed bins over [0,14]. Border fraction separates
exact zeros. Out-of-range finite values clip into the endpoint bins. Constant
and entirely missing columns are supported. Custom fixed edges can override
these defaults. Bin edges never use test labels, predictions or attributes.

Default selection processes consecutive pools of eight baseline batches.
Each pool runs 64 rounds with 128 candidate same-class swaps per round. The
best strictly improving proposal is accepted. Local RNG seed is
88111 + epoch*10000 + pool_id. An image occurrence moves only to an existing
slot of the same class. No image is added, removed, edited or reweighted.

Default weights are 1/A. `weights=[...]` supplies normalized fixed weights.
`source_difference` weights by the source whole-epoch Real/Fake histogram
distance divided by s(a); `capped_source_difference` additionally caps weights.
These automatic modes are experimental ablations, not recommended evidence
of universal improvement. The equal-weight default does not adapt to test data.

## Attribute measurement contract

Record units, extraction software/version, missing-value policy and image
preprocessing in your experiment configuration. Example six-column schema:

| Column | Definition |
| --- | --- |
| face_area | bounding-box area divided by original image area |
| face_y | face center Y divided by original image height |
| abs_yaw | absolute head yaw in degrees |
| log2_pixel_area | log2(original width * original height) |
| sharpness | log(1 + variance of grayscale Laplacian), at a declared resize |
| border_black_fraction | declared border-region black-pixel fraction |

Face size may instead use normalized inter-pupil distance; position may use
X and Y jointly; pose may use yaw, pitch and roll. Multiple columns of one
category change its effective weight unless weights are adjusted accordingly.
Native geometry proxies and post-augmentation image attributes must not be
silently conflated. If measuring epoch-specific augmented attributes, replay
the exact same per-image transform in both arms and pass that matrix to
`align(..., attributes=epoch_attributes)`. Keep the fitted bins/scales frozen.

## Matched experiment

1. Save source split IDs, labels and attribute order; fit only on training.
2. Produce the baseline epoch plan once; keep its exact occurrences in alignment.
3. Use the same initialization, transforms, optimizer/schedule, batch size,
   epoch count, mixed precision and checkpoint selection rule.
4. Log per-batch d(b,a), its epoch sum, the scaled joint objective and undefined
   batch count. Compare against the exact pre-swap plan.
5. Select checkpoints only on a fixed source validation protocol, then evaluate
   the same held-out datasets using identical frame/video aggregation and
   threshold. Report AUC, accuracy and balanced accuracy with units specified.
6. Repeat matched seeds before making robustness or significance claims.

Individual batch distances may increase as the pool sum improves. Optimizing
several marginal histograms does not match the complete joint attribute
distribution. Attribute measurement errors and semantic/forgery correlations
can make some alignment objectives harmful.
