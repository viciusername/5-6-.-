from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from .utils import ensure_dir, load_and_clean_car_data, save_text


@dataclass
class Lab6Outputs:
    out_dir: Path


def _ordinal_encode(X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Encode all columns to numeric (ordinal). Used for KMeans + pairwise plots."""
    Xc = X.copy()
    cat_cols = [c for c in Xc.columns if not pd.api.types.is_numeric_dtype(Xc[c])]
    num_cols = [c for c in Xc.columns if c not in cat_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(Xc[cat_cols]) if cat_cols else np.empty((len(Xc), 0))
    X_num = Xc[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(Xc), 0))

    X_all = np.concatenate([X_cat, X_num], axis=1)
    feature_names = cat_cols + num_cols
    return X_all, feature_names


def _plot_pairs_with_centroids(Xn: np.ndarray, feature_names: List[str], centroids: np.ndarray, pairs: List[Tuple[int, int]], out_dir: Path) -> None:
    for idx, (i, j) in enumerate(pairs, start=1):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(Xn[:, i], Xn[:, j], s=12, alpha=0.6)
        ax.scatter(centroids[:, i], centroids[:, j], s=150, marker="X")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(f"KMeans clusters: {feature_names[i]} vs {feature_names[j]}")
        plt.tight_layout()
        plt.savefig(out_dir / f"pairplot_{idx}_{feature_names[i]}_{feature_names[j]}.png", dpi=200)
        plt.close(fig)


def run_lab6(data_csv: str | Path, outputs_root: str | Path = "outputs") -> Lab6Outputs:
    out_dir = ensure_dir(Path(outputs_root) / "lab6")

    X, y, _ = load_and_clean_car_data(data_csv)
    Xn, feature_names = _ordinal_encode(X)

    # 2) KMeans with k = number of classes
    k = int(pd.Series(y).nunique())
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    clusters = km.fit_predict(Xn)

    # 3) 4 pairwise plots + centroids
    p = Xn.shape[1]
    base_pairs = [(0, 1), (2, 3), (4, 5), (0, 5)]
    pairs = [(a % p, b % p) for a, b in base_pairs if p > 0]
    _plot_pairs_with_centroids(Xn, feature_names, km.cluster_centers_, pairs, out_dir)

    # 4) Cluster sizes + crosstab with true classes
    pd.Series(clusters).value_counts().sort_index().to_csv(out_dir / "cluster_sizes.csv")
    pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(y, name="class")).to_csv(out_dir / "cluster_vs_class_table.csv")

    # 5) Optimal k by 4 metrics
    ks = list(range(2, 11))

    def compute_metrics(Xmat: np.ndarray) -> pd.DataFrame:
        rows = []
        for kk in ks:
            model = KMeans(n_clusters=kk, n_init=20, random_state=42)
            lbl = model.fit_predict(Xmat)
            inertia = float(model.inertia_)
            sil = float(silhouette_score(Xmat, lbl))
            ch = float(calinski_harabasz_score(Xmat, lbl))
            db = float(davies_bouldin_score(Xmat, lbl))
            rows.append({"k": kk, "inertia": inertia, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db})
        return pd.DataFrame(rows)

    # 6) Scaling impact
    scalers: Dict[str, object | None] = {"none": None, "standard": StandardScaler(), "minmax": MinMaxScaler()}
    tables: Dict[str, pd.DataFrame] = {}
    for name, scaler in scalers.items():
        Xm = Xn if scaler is None else scaler.fit_transform(Xn)
        dfm = compute_metrics(Xm)
        dfm.to_csv(out_dir / f"metrics_{name}.csv", index=False)
        tables[name] = dfm

    def plot_metric(metric: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, dfm in tables.items():
            ax.plot(dfm["k"], dfm[metric], marker="o", label=name)
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs k (вплив масштабування)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close(fig)

    plot_metric("inertia", "metric_inertia_vs_k.png")
    plot_metric("silhouette", "metric_silhouette_vs_k.png")
    plot_metric("calinski_harabasz", "metric_calinski_harabasz_vs_k.png")
    plot_metric("davies_bouldin", "metric_davies_bouldin_vs_k.png")

    note = []
    note.append(f"Кількість класів у ЛР5: {k}. У ЛР6 базова кластеризація виконана k-means з k={k}.")
    note.append("Метрики: inertia (min), silhouette (max), Calinski-Harabasz (max), Davies-Bouldin (min).")
    save_text(out_dir / "note_k_selection.txt", "\n".join(note))

    return Lab6Outputs(out_dir=out_dir)
