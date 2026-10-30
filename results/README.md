# Xception observations

`xception.csv` records the study's reported single-seed (seed 0) summaries.
Values are percentages; differences are percentage points. Results are macro
averages across five unseen test sets: Celeb-DF-v2, DFDC subset, DFDC Preview
subset, WildDeepfake and DFD. FF++ is the source training dataset; its test
split is excluded from this external average. Use the same dataset subset
manifests to reproduce a numerical average; dataset names alone do not identify
those subsets.

The initial sharpness experiment is a separate historical run; its accuracy
should not be treated as a matched repeat of the later six-attribute sweep.
The six single-category sweep and multi-attribute ablations share the reported
B0 reference. Source validation selects the checkpoints; external scores are
not used for checkpoint or weight fitting.

The source multi-attribute summary reports:

| Method | Selected epoch | Image AUC | Accuracy | Balanced accuracy | Video AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 6 | 72.127857 | 60.118610 | 64.185811 | 76.210848 |
| Equal six | 3 | 73.520619 | 62.020444 | 65.371051 | 77.600399 |
| Source difference | 10 | 70.290666 | 62.939055 | 63.752351 | 72.949266 |

Equal weighting reduced the cumulative scaled joint histogram objective by
29.5480%; source-difference weighting reduced it by 33.3434%. These values
describe the joint objective, **not** every individual attribute's raw L2.
Despite larger objective reduction, automatic weighting harmed image AUC.
Aligning face size alone also reduced AUC. Consequently, matching nuisance
attribute distributions is an optimization property, not proof of improved
generalization or a causal shortcut diagnosis.

This release provides the batch-selection implementation and recorded results,
not a complete Xception detector reproduction package. Training images,
private manifests, detector weights, per-image predictions and full training
logs are not redistributed. Exact historical model metrics require those
artifacts and the original detector pipeline. The runnable examples reproduce
the selection mechanism on synthetic inputs only. Multiple seeds and
uncertainty intervals remain necessary for stronger scientific claims.
