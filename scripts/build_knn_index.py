"""Bangun index k-NN Jaccard dan ukur apakah ia bisa menggantikan UMAP saat inference.

umap_model.pkl 77 MB dan umap.transform() menjalankan pencarian tetangga terhadap
graph 23k baris tiap request -- terlalu berat untuk hosting gratis. UMAP di-fit
dengan metric='jaccard', jadi voting k-NN Jaccard langsung di ruang biner 186-dim
seharusnya mereproduksi keputusan cluster yang sama.

Script ini membangun indexnya lalu memvalidasi klaim itu dengan leave-one-out
terhadap label asli di cluster_assignments.csv.

Jalankan: python scripts/build_knn_index.py
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("DEVPATH_OUT_DIR", os.path.join(BASE, "Outputs"))
K = 25


def build_matrix():
    col_map = json.load(open(os.path.join(OUT, "artifacts", "col_feature_map.json")))
    feature_names = json.load(open(os.path.join(OUT, "artifacts", "feature_names.json")))
    labels = pd.read_csv(os.path.join(OUT, "models", "cluster_assignments.csv"))

    parts = [pd.read_csv(os.path.join(BASE, "Dataset", f"part_{i}_SO_survey.csv"),
                         low_memory=False) for i in range(1, 7)]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("ResponseId", keep="first")
    df = labels[["ResponseId", "cluster_id"]].merge(df, on="ResponseId")  # urutan ikut labels
    assert len(df) == len(labels), f"join hilang baris: {len(df)} != {len(labels)}"

    cols = []
    for col, feats in col_map.items():
        skills = [f.split("__", 1)[1] for f in feats]
        cols.append(df[col].fillna("").str.get_dummies(sep=";")
                    .reindex(columns=skills, fill_value=0).to_numpy(np.uint8))
    X = np.hstack(cols)
    assert X.shape[1] == len(feature_names), f"{X.shape[1]} != {len(feature_names)}"
    return X, df.cluster_id.to_numpy(np.int8)


def jaccard_sim(Q, X, x_sizes):
    """Similaritas Jaccard tiap baris Q terhadap seluruh X."""
    inter = Q.astype(np.float32) @ X.astype(np.float32).T
    union = Q.sum(1, dtype=np.float32)[:, None] + x_sizes[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def vote(sims, labels, k, n_clusters):
    """Voting k tetangga terdekat, berbobot similaritas. -> (cluster, skor per cluster)"""
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    rows = np.arange(len(sims))[:, None]
    w, lab = sims[rows, idx], labels[idx]
    scores = np.zeros((len(sims), n_clusters), np.float32)
    for c in range(n_clusters):
        scores[:, c] = (w * (lab == c)).sum(1)
    total = scores.sum(1, keepdims=True)
    return scores.argmax(1), scores / np.maximum(total, 1e-9)


def main():
    X, y = build_matrix()
    n_clusters = int(y.max()) + 1
    print(f"Matriks    : {X.shape}, {X.nbytes/1024**2:.1f} MB dense")

    x_sizes = X.sum(1, dtype=np.float32)
    correct = 0
    for s in range(0, len(X), 1000):
        sims = jaccard_sim(X[s:s + 1000], X, x_sizes)
        sims[np.arange(len(sims)), np.arange(s, s + len(sims))] = -1.0  # buang diri sendiri
        pred, _ = vote(sims, y, K, n_clusters)
        correct += (pred == y[s:s + len(sims)]).sum()
        print(f"  ...{s + len(sims):,}/{len(X):,}", end="\r")

    acc = correct / len(X)
    print(f"\nLeave-one-out agreement (k={K}): {acc*100:.2f}%  "
          f"({correct:,}/{len(X):,})")

    path = os.path.join(OUT, "artifacts", "knn_index.npz")
    np.savez_compressed(path, packed=np.packbits(X, axis=1), n_features=X.shape[1], labels=y)
    print(f"Ditulis    : {path} ({os.path.getsize(path)/1024:.0f} KB)")
    print("Ganti UMAP dengan k-NN." if acc >= 0.95 else "Agreement rendah -- pertahankan UMAP.")


if __name__ == "__main__":
    main()
