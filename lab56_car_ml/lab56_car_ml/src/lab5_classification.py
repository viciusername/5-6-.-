from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from .utils import ensure_dir, load_and_clean_car_data, save_text


@dataclass
class Lab5Outputs:
    out_dir: Path


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.2,
    )
    return pre


def _scatter_matrix(X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    # Scatter matrix works only for numeric features; we ordinalize categories for visualization.
    X_vis = X.copy()
    for col in X_vis.columns:
        if not pd.api.types.is_numeric_dtype(X_vis[col]):
            X_vis[col] = X_vis[col].astype("category").cat.codes

    # Sample for speed
    n = len(X_vis)
    take = min(500, n)
    if take < n:
        idx = X_vis.sample(n=take, random_state=42).index
        X_vis = X_vis.loc[idx]
        y = y.loc[idx]

    df_vis = X_vis.copy()
    df_vis["__label__"] = y.astype("category").cat.codes

    pd.plotting.scatter_matrix(
        df_vis.drop(columns=["__label__"]),
        figsize=(10, 10),
        diagonal="hist",
        alpha=0.6,
    )
    plt.suptitle("Scatter-matrix (категорії закодовані числами для візуалізації)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Own implementation (Task 10)
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def pick_three_per_class(y: pd.Series, random_state: int = 42) -> List[int]:
    rng = np.random.default_rng(random_state)
    idxs: List[int] = []
    for cls in sorted(y.unique()):
        cls_idx = np.where(y.to_numpy() == cls)[0]
        if len(cls_idx) == 0:
            continue
        choose = cls_idx if len(cls_idx) <= 3 else rng.choice(cls_idx, size=3, replace=False)
        idxs.extend(list(choose))
    return idxs


def run_lab5(data_csv: str | Path, outputs_root: str | Path = "outputs") -> Lab5Outputs:
    out_dir = ensure_dir(Path(outputs_root) / "lab5")

    X, y, info = load_and_clean_car_data(data_csv)

    # 1) Basic stats
    X.describe(include="all").to_csv(out_dir / "describe.csv")

    stats_text = []
    stats_text.append(f"Rows (raw): {info['n_rows_raw']}")
    stats_text.append(f"Columns: {info['n_cols_raw']}  => {info['columns']}")
    stats_text.append(f"Label column: {info['label_col']}")
    stats_text.append(f"Rows with missing values: {info['missing_rows']}")
    stats_text.append(f"Duplicate rows: {info['duplicate_rows']}")
    stats_text.append(f"Rows (clean): {info['n_rows_clean']}")
    stats_text.append("Class distribution (clean):")
    for k, v in info["class_counts"].items():
        stats_text.append(f"  {k}: {v}")
    save_text(out_dir / "dataset_stats.txt", "\n".join(stats_text))

    # 2) Scatter-matrix
    _scatter_matrix(X, y, out_dir / "scatter_matrix.png")

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = _make_preprocessor(X)

    # 4) Classifier (choose KNN as base)
    knn_base = Pipeline([("pre", pre), ("clf", KNeighborsClassifier(n_neighbors=7))])
    knn_scaled = Pipeline([
        ("pre", pre),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ])

    def eval_and_save(pipe: Pipeline, tag: str) -> Dict[str, Any]:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        save_text(out_dir / f"classification_report_{tag}.txt", classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, xticks_rotation=45, values_format='d')
        ax.set_title(f"Confusion matrix ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=200)
        plt.close(fig)

        return {
            "tag": tag,
            "accuracy": float((y_pred == y_test).mean()),
        }

    res_base = eval_and_save(knn_base, "knn_no_scaling")
    res_scaled = eval_and_save(knn_scaled, "knn_with_scaling")
    pd.DataFrame([res_base, res_scaled]).to_csv(out_dir / "scaling_comparison.csv", index=False)

    # 8) Model/parameter selection to maximize test accuracy (FAST holdout validation)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    candidates: List[Tuple[str, Pipeline]] = [
        ("knn_k7", Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=False)), ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))])),
        ("knn_k11", Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=False)), ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance"))])),
        ("logreg_C1", Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=500, C=1.0))])),
        ("logreg_C2", Pipeline([("pre", pre), ("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=500, C=2.0))])),
        ("rf_200", Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None))])),
        ("rf_depth12", Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42, n_estimators=200, max_depth=12))])),
    ]

    rows = []
    best_estimator: Pipeline | None = None
    best_val = -1.0
    best_name = None
    for name, pipe in candidates:
        pipe.fit(X_tr, y_tr)
        val_pred = pipe.predict(X_val)
        val_acc = float((val_pred == y_val).mean())
        rows.append({"model": name, "val_accuracy": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_name = name
            best_estimator = pipe

    assert best_estimator is not None
    pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).to_csv(out_dir / "model_selection.csv", index=False)

    # Fit best on full train, evaluate on test
    best_estimator.fit(X_train, y_train)
    y_pred_best = best_estimator.predict(X_test)
    save_text(out_dir / "best_model_classification_report.txt", classification_report(y_test, y_pred_best))

    cm = confusion_matrix(y_test, y_pred_best, labels=sorted(y.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, xticks_rotation=45, values_format='d')
    ax.set_title(f"Confusion matrix (best model: {best_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_best_model.png", dpi=200)
    plt.close(fig)

    # 10) Own Euclidean distance + nearest neighbor check
    steps = best_estimator.named_steps
    pre_step = steps.get("pre")
    scaler_step = steps.get("scaler")

    Xtr_t = pre_step.transform(X_train)
    Xte_t = pre_step.transform(X_test)
    if scaler_step is not None:
        Xtr_t = scaler_step.fit_transform(Xtr_t)
        Xte_t = scaler_step.transform(Xte_t)

    if hasattr(Xtr_t, "toarray"):
        Xtr_dense = Xtr_t.toarray()
    else:
        Xtr_dense = np.asarray(Xtr_t)
    if hasattr(Xte_t, "toarray"):
        Xte_dense = Xte_t.toarray()
    else:
        Xte_dense = np.asarray(Xte_t)

    ytr = y_train.reset_index(drop=True)
    yte = y_test.reset_index(drop=True)

    idx_pick = pick_three_per_class(yte, random_state=42)
    results = []
    for i in idx_pick:
        vec = Xte_dense[i]
        # find nearest neighbor in train
        dists = np.sqrt(np.sum((Xtr_dense - vec) ** 2, axis=1))
        j = int(np.argmin(dists))
        results.append({
            "test_index": int(i),
            "test_class": str(yte.iloc[i]),
            "nearest_train_index": int(j),
            "nearest_train_class": str(ytr.iloc[j]),
            "same_class": bool(yte.iloc[i] == ytr.iloc[j]),
            "distance": float(dists[j]),
        })

    pd.DataFrame(results).to_csv(out_dir / "nearest_neighbor_check.csv", index=False)

    concl = []
    concl.append("Перевірка методу k-NN через власну евклідову відстань (nearest neighbor):")
    same_ratio = float(np.mean([r["same_class"] for r in results])) if results else float('nan')
    concl.append(f"Частка збігів класу між тестовим об'єктом і найближчим сусідом: {same_ratio:.3f}")
    concl.append("Якщо частка висока, k-NN підходить для цього набору; якщо низька — ознаки погано розділяють класи або потрібне інше кодування/масштабування.")
    save_text(out_dir / "knn_neighbor_conclusion.txt", "\n".join(concl))

    return Lab5Outputs(out_dir=out_dir)
