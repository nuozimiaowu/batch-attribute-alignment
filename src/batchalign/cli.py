"""Source-training CSV to reproducible batch plans."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import BatchAligner, balanced_plan, shuffled_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", type=Path, help="training CSV: id,label and attribute columns"
    )
    parser.add_argument("--attributes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, help="existing baseline .npy plan; indices follow CSV rows"
    )
    parser.add_argument(
        "--balanced", action="store_true", help="generate a 1:1 example baseline"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline-seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=88111)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--proposals", type=int, default=128)
    parser.add_argument("--pool-batches", type=int, default=8)
    parser.add_argument(
        "--fit-state",
        type=Path,
        help="reuse fitted training rules, e.g. for later epochs",
    )
    args = parser.parse_args(argv)
    try:
        with args.csv.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"id", "label", *args.attributes}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain: {sorted(required)}")
            rows = list(reader)
        ids = [row["id"] for row in rows]
        if not rows or len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError("CSV IDs must be nonempty and unique")
        labels = np.array([int(row["label"]) for row in rows])
        values = np.array(
            [
                [float(row[a]) if row[a].strip() else np.nan for a in args.attributes]
                for row in rows
            ]
        )
        if args.plan and args.balanced:
            raise ValueError("--plan and --balanced are mutually exclusive")
        plan = (
            np.load(args.plan, allow_pickle=False)
            if args.plan
            else (balanced_plan if args.balanced else shuffled_plan)(
                labels, args.batch_size, args.baseline_seed
            )
        )
        if args.fit_state:
            aligner = BatchAligner.load(args.fit_state)
            if tuple(args.attributes) != aligner.attribute_names_ or not np.array_equal(
                labels, aligner.labels_
            ):
                raise ValueError(
                    "fitted state requires identical attribute columns and label row ordering"
                )
        else:
            if args.epoch != 0:
                raise ValueError(
                    "fit on epoch 0 first; supply --fit-state for later epochs"
                )
            aligner = BatchAligner(
                seed=args.seed,
                rounds=args.rounds,
                proposals=args.proposals,
                pool_batches=args.pool_batches,
            ).fit(values, labels, plan, args.attributes)
        # Avoid silently overwriting a previous experiment's artifacts.
        args.output.mkdir(parents=True, exist_ok=False)
        result = aligner.align(plan, values, args.epoch)
        result.save(args.output)
        aligner.save(args.output / "fit.npz")
        with (args.output / "row_mapping.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["row", "id", "label"])
            writer.writerows((i, name, int(labels[i])) for i, name in enumerate(ids))
        (args.output / "command.json").write_text(
            json.dumps(
                {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(result.summary(), indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
