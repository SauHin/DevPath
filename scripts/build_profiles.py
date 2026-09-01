"""Hitung prevalensi seluruh 186 skill per cluster, dari data yang sudah ada di repo.

cluster_profiles.json hanya menyimpan 8 skill teratas per cluster, jadi skill gap
mentok di 8 kandidat. Script ini menghitung ulang untuk semua fitur, plus sinyal
WantToWorkWith yang membedakan roadmap dari skill gap. Tidak butuh Colab dan
tidak retraining -- label cluster diambil apa adanya dari cluster_assignments.csv.

Jalankan: python scripts/build_profiles.py
"""
import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("DEVPATH_OUT_DIR", os.path.join(BASE, "Outputs"))


def load_survey():
    parts = [pd.read_csv(os.path.join(BASE, "Dataset", f"part_{i}_SO_survey.csv"),
                         low_memory=False) for i in range(1, 7)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("ResponseId", keep="first")


def main():
    col_map = json.load(open(os.path.join(OUT, "artifacts", "col_feature_map.json")))
    feature_names = json.load(open(os.path.join(OUT, "artifacts", "feature_names.json")))
    labels = pd.read_csv(os.path.join(OUT, "models", "cluster_assignments.csv"))

    df = load_survey().merge(labels[["ResponseId", "cluster_id"]], on="ResponseId")
    assert len(df) == len(labels), f"join hilang baris: {len(df)} != {len(labels)}"
    print(f"Responden  : {len(df):,}  |  cluster: {sorted(df.cluster_id.unique())}")

    prevalence = {int(c): {} for c in sorted(df.cluster_id.unique())}
    for col, feats in col_map.items():
        skills = [f.split("__", 1)[1] for f in feats]
        have = df[col].fillna("").str.get_dummies(sep=";").reindex(columns=skills, fill_value=0)
        want = df[col.replace("HaveWorkedWith", "WantToWorkWith")] \
            .fillna("").str.get_dummies(sep=";").reindex(columns=skills, fill_value=0)
        # "ingin" hanya dihitung untuk yang belum memakainya
        want_new = want.where(have == 0, 0)

        g = df.cluster_id
        have_pct = have.groupby(g).mean()
        # penyebut = anggota cluster yang belum punya skill tsb
        want_pct = want_new.groupby(g).sum() / (1 - have).groupby(g).sum().replace(0, pd.NA)

        for cid in prevalence:
            for feat, skill in zip(feats, skills):
                prevalence[cid][feat] = {
                    "have": round(float(have_pct.loc[cid, skill]), 4),
                    "want": round(float(want_pct.loc[cid, skill] or 0), 4),
                }

    for cid, feats in prevalence.items():
        assert len(feats) == len(feature_names), f"cluster {cid}: {len(feats)} != {len(feature_names)}"

    path = os.path.join(OUT, "artifacts", "cluster_prevalence.json")
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in prevalence.items()}, f, separators=(",", ":"))
    print(f"Ditulis    : {path} ({os.path.getsize(path)/1024:.0f} KB, "
          f"{len(prevalence)} cluster x {len(feature_names)} fitur)")

    for cid in sorted(prevalence):
        top = sorted(prevalence[cid].items(), key=lambda kv: -kv[1]["want"])[:3]
        print(f"  cluster {cid} paling diminati: " +
              ", ".join(f"{k.split('__',1)[1]} {v['want']*100:.0f}%" for k, v in top))


if __name__ == "__main__":
    main()
