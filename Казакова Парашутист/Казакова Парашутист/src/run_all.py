"""Entry point: generate all artifacts for Lab #2.

Usage:
  python -m src.run_all

Outputs are written to ./outputs
"""

from __future__ import annotations

import os

from .experiments import run_all_experiments


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    out_dir = os.path.abspath(out_dir)

    # 1) Flowchart
    # Keep in docs/ to separate, but write the png into outputs/
    from docs.flowchart_generator import main as make_flowchart

    make_flowchart(out_dir)

    # 2) Experiments + plots + tables
    run_all_experiments(out_dir)

    print("Готово! Результати у папці:", out_dir)


if __name__ == "__main__":
    main()
