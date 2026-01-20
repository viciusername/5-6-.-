from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_and_clean_car_data(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load Car Evaluation dataset, clean, and return (X, y, info).

    Expected columns (may vary by file):
    - Features: PurchasePrice, MaintenanceCost, Doors, Passengers, Cargo, Safety
    - Label: Acceptability

    Cleaning requirements (per assignment):
    - count missing rows, drop them
    - count duplicates, drop them
    - replace 'Interval[{5, Infinity}]' with 5
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    info: Dict[str, Any] = {
        "n_rows_raw": int(len(df)),
        "n_cols_raw": int(df.shape[1]),
        "columns": list(df.columns),
    }

    # Replace the special string if present
    df = df.replace({"Interval[{5, Infinity}]": 5, "Interval[{5,Infinity}]": 5})

    # Missing
    missing_rows = int(df.isna().any(axis=1).sum())
    info["missing_rows"] = missing_rows
    if missing_rows:
        df = df.dropna(axis=0).reset_index(drop=True)

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    info["duplicate_rows"] = dup_rows
    if dup_rows:
        df = df.drop_duplicates().reset_index(drop=True)

    info["n_rows_clean"] = int(len(df))

    # Identify label column
    label_col = None
    for cand in ["Acceptability", "class", "label", "target"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]
    info["label_col"] = label_col

    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    # Try to coerce numeric-looking columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="ignore")

    info["class_counts"] = y.value_counts().to_dict()
    return X, y, info
