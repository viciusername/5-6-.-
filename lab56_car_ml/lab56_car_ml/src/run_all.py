from __future__ import annotations

from pathlib import Path

from .lab5_classification import run_lab5
from .lab6_clustering import run_lab6


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_csv = root / "data" / "Car evaluation.csv"
    outputs_root = root / "outputs"

    run_lab5(data_csv, outputs_root=outputs_root)
    run_lab6(data_csv, outputs_root=outputs_root)

    print("Done. See outputs/.")


if __name__ == "__main__":
    main()
