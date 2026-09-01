"""Pilih satu stack contoh yang nyata untuk tiap cluster.

Kartu "See an example result" di landing page dulu menyusun contohnya dari
top_skills (8 skill dengan prevalensi tertinggi). Prevalensi mentah didominasi
skill yang umum di mana-mana, jadi contoh untuk cluster 2 (JavaScript, Docker,
Python, npm, HTML/CSS, VS Code, SQL, TypeScript) justru terprediksi ke cluster 5.

Script ini mengambil responden asli dari tiap cluster, jadi contohnya benar
menurut konstruksi, lalu memverifikasinya lewat model yang sama dengan app.py
dan turun ke kandidat berikutnya kalau ada yang meleset.

Jalankan: python scripts/build_examples.py   (setelah retraining, jalankan ulang)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("DEVPATH_OUT_DIR", os.path.join(BASE, "Outputs"))
ART = os.path.join(OUT, "artifacts")

COL_ORDER = ["LanguageHaveWorkedWith", "WebframeHaveWorkedWith", "DatabaseHaveWorkedWith",
             "PlatformHaveWorkedWith", "DevEnvsHaveWorkedWith", "AIModelsHaveWorkedWith"]

# Jumlah skill tidak bisa dipatok satu angka: median cluster 2 adalah 42 skill
# sementara cluster lain 20-23, karena "banyak tooling" justru yang mendefinisikan
# cluster itu. Contoh dengan 8 tag akan tampak seperti generalist dan terprediksi
# ke cluster 5. Jadi bandnya relatif terhadap median clusternya sendiri.
# ponytail: band +-25% dari median, cukup untuk 6 cluster ini; kalau retraining
# menghasilkan cluster yang lebih tipis, longgarkan bandnya atau naikkan kandidat.
BAND, N_CANDIDATES = 0.25, 60


def load_survey():
    parts = [pd.read_csv(os.path.join(BASE, "Dataset", f"part_{i}_SO_survey.csv"),
                         low_memory=False) for i in range(1, 7)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("ResponseId", keep="first")


def main():
    prevalence = {int(k): v for k, v in
                  json.load(open(os.path.join(ART, "cluster_prevalence.json"), encoding="utf-8")).items()}
    mlb = joblib.load(os.path.join(ART, "mlb_encoders.pkl"))
    umap_model = joblib.load(os.path.join(ART, "umap_model.pkl"))
    kmeans = joblib.load(os.path.join(ART, "kmeans_final.pkl"))
    labels = pd.read_csv(os.path.join(OUT, "models", "cluster_assignments.csv"))

    df = load_survey().merge(labels[["ResponseId", "cluster_id"]], on="ResponseId")
    assert len(df) == len(labels), f"join hilang baris: {len(df)} != {len(labels)}"

    def skills_of(row):
        out = {}
        for col in COL_ORDER:
            picked = [s for s in str(row[col]).split(";") if s and s in set(mlb[col].classes_)]
            if picked:
                out[col] = sorted(picked)
        return out

    def encode(skills):
        return np.hstack([mlb[col].transform([skills.get(col, [])]).astype(np.float32)
                          for col in COL_ORDER])

    examples = {}
    for cid in sorted(prevalence):
        prev = prevalence[cid]
        members = df[df.cluster_id == cid]

        counts = members.apply(lambda r: sum(len(v) for v in skills_of(r).values()), axis=1)
        lo, hi = counts.median() * (1 - BAND), counts.median() * (1 + BAND)

        scored = []
        for _, row in members.iterrows():
            s = skills_of(row)
            n = sum(len(v) for v in s.values())
            if not (lo <= n <= hi) or not s.get("LanguageHaveWorkedWith"):
                continue
            feats = [f"{p}__{name}" for col, p in zip(COL_ORDER, ("lang", "wf", "db", "plat", "env", "ai"))
                     for name in s.get(col, [])]
            # Setipikal mungkin: rata-rata prevalensi skill-nya di cluster sendiri.
            scored.append((sum(prev.get(f, {}).get("have", 0) for f in feats) / n, s))
        scored.sort(key=lambda x: -x[0])

        cands = [s for _, s in scored[:N_CANDIDATES]]
        if not cands:
            raise SystemExit(f"cluster {cid}: tidak ada kandidat dalam band ukuran")

        # Satu transform untuk semua kandidat, lalu ambil yang paling yakin di
        # antara yang terprediksi balik ke clusternya. Confidence dihitung sama
        # seperti app.py: softmax atas jarak negatif ke tiap centroid.
        reduced = umap_model.transform(np.vstack([encode(s) for s in cands]))
        got = kmeans.predict(reduced)
        dist = kmeans.transform(reduced)
        conf = np.exp(-dist) / np.exp(-dist).sum(axis=1, keepdims=True)

        hits = [(conf[i][cid], cands[i]) for i in range(len(cands)) if got[i] == cid]
        if not hits:
            raise SystemExit(f"cluster {cid}: tidak ada kandidat yang terprediksi balik ke clusternya")

        best, s = max(hits, key=lambda h: h[0])
        examples[cid] = s
        n = sum(len(v) for v in s.values())
        print(f"cluster {cid}: {n} skill, confidence {best:.2f} "
              f"({len(hits)}/{len(cands)} kandidat cocok) -> {sorted(sum(s.values(), []))}")

    path = os.path.join(ART, "cluster_examples.json")
    json.dump({str(k): v for k, v in examples.items()},
              open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nDitulis: {path}")


if __name__ == "__main__":
    main()
