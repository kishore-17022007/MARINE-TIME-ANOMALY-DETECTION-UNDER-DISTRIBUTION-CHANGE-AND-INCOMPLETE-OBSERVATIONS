from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.synthetic_benchmark import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved maritime anomaly models on controlled synthetic cases.")
    parser.add_argument("--experiments", nargs="+", choices=("1-day", "7-day"), default=("1-day", "7-day"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    args = parser.parse_args()

    results = []
    for experiment in args.experiments:
        result = run_experiment(ROOT / "experiments" / experiment, tuple(args.seeds))
        results.append(result)
        print(f"{experiment}: {result['summary_csv']} and {result['metrics_csv']}", flush=True)


if __name__ == "__main__":
    main()