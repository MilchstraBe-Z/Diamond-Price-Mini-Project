"""Command-line entry point: ``diamonds train [--csv PATH] [--output PATH]``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from diamonds.data import clean, load_data
from diamonds.features import encode_ordinal
from diamonds.model import evaluate, feature_importances, split, train_baseline, train_forest

TOP_FEATURES = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diamonds", description="Diamond price prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Clean the data, train both models, report.")
    train.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Local CSV with the diamonds columns. Default: download from seaborn.",
    )
    train.add_argument(
        "--output",
        type=Path,
        default=Path("report.json"),
        help="Where to write the JSON report (default: report.json).",
    )
    return parser


def run_training(csv_path: Path | None) -> dict:
    """Run the whole pipeline and return the report as a plain dict."""
    raw = load_data(csv_path)
    cleaned, removed = clean(raw)
    X_train, X_test, y_train, y_test = split(encode_ordinal(cleaned))

    baseline = train_baseline(X_train, y_train)
    forest = train_forest(X_train, y_train)
    importances = feature_importances(forest)

    return {
        "data": {
            "source": str(csv_path) if csv_path else "seaborn:diamonds",
            "rows_raw": len(raw),
            "rows_clean": len(cleaned),
            "rows_removed": removed,
            "rows_train": len(X_train),
            "rows_test": len(X_test),
        },
        "metrics": {
            "baseline": evaluate(baseline, X_test, y_test),
            "forest": evaluate(forest, X_test, y_test),
        },
        "top_features": dict(list(importances.items())[:TOP_FEATURES]),
    }


def format_table(report: dict) -> str:
    lines = [
        f"{'model':<10}{'test R2':>10}{'test MAE (USD)':>17}",
        "-" * 37,
    ]
    for name, scores in report["metrics"].items():
        lines.append(f"{name:<10}{scores['r2']:>10.3f}{scores['mae']:>17,.0f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "train":
        try:
            report = run_training(args.csv)
        except (FileNotFoundError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1

        data = report["data"]
        print(
            f"Data: {data['rows_raw']} rows -> {data['rows_clean']} after cleaning "
            f"{data['rows_removed']}; {data['rows_test']} test rows"
        )
        print(format_table(report))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
