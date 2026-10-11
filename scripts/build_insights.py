"""Grafik dan tabel insight untuk bagian "Hasil Analisis Data" di README.

Semua angka dihitung ulang dari data yang sudah ada di repo: Dataset/ untuk
kualitas data, Outputs/models/dataset_with_clusters.csv untuk profil persona,
dan Outputs/artifacts/cluster_prevalence.json untuk minat skill. Tidak butuh
Colab dan tidak retraining.

Jalankan: python scripts/build_insights.py   (setelah retraining, jalankan ulang)
Hasil   : docs/insights/*.png, plus tabel markdown di stdout untuk README.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from build_profiles import BASE, OUT, load_survey

DOCS = os.path.join(BASE, "docs", "insights")
SKILL_COLS = ["LanguageHaveWorkedWith", "WebframeHaveWorkedWith", "DatabaseHaveWorkedWith",
              "PlatformHaveWorkedWith", "DevEnvsHaveWorkedWith", "AIModelsHaveWorkedWith"]
# skill yang paling diminati lintas persona (lihat build_profiles.py)
DEMAND_SKILLS = ["lang__Rust", "lang__Go", "db__PostgreSQL", "plat__Kubernetes",
                 "lang__TypeScript", "lang__Python"]
MIN_COUNTRY_N = 30  # negara dengan < 30 data gaji terlalu kecil untuk median yang stabil
SALARY_RANGE = (5000, 500000)  # filter outlier yang sama dengan profile_clusters di notebook 03

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "font.size": 10,
})


def short(label):
    return label.replace(" Developer", "")


def id_num(x, d=0):
    """Format angka Indonesia: titik untuk ribuan, koma untuk desimal."""
    return f"{x:,.{d}f}".translate(str.maketrans(",.", ".,"))


def salary(d):
    return d.ConvertedCompYearly.where(d.ConvertedCompYearly.between(*SALARY_RANGE, inclusive="neither"))


def clean_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def data_quality(raw):
    """Missing rate per kolom skill dan corong cleaning, aturan sama dengan notebook 02."""
    missing = (raw[SKILL_COLS].isna().mean() * 100).sort_values()
    has_lang = raw[raw.LanguageHaveWorkedWith.notna()]
    other_core = SKILL_COLS[1:5]
    kept = has_lang[has_lang[other_core].isna().sum(axis=1) < 2]
    funnel = [("Data mentah", len(raw)), ("Mengisi bahasa\npemrograman", len(has_lang)),
              ("Maks. 1 kolom inti\nkosong (dipakai)", len(kept))]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a.barh([c.replace("HaveWorkedWith", "") for c in missing.index], missing.values,
           color=BLUE, height=0.6)
    for i, v in enumerate(missing.values):
        a.text(v + 1, i, f"{id_num(v, 1)}%", va="center", color=INK_2)
    a.set_xlim(0, 80)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a.set_title("Persentase responden yang tidak mengisi kolom skill")
    clean_axes(a)

    names, counts = zip(*funnel)
    b.bar(names, counts, color=[BLUES[2], BLUES[3], BLUES[5]], width=0.6)
    for i, n in enumerate(counts):
        b.text(i, n + 700, f"{id_num(n)}\n({id_num(n / counts[0] * 100, 1)}%)", ha="center", color=INK)
    b.set_ylim(0, counts[0] * 1.2)
    b.set_yticks([])
    b.set_title("Jumlah responden di tiap tahap cleaning")
    for side in ("top", "right", "left"):
        b.spines[side].set_visible(False)
    b.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "data_quality.png"), dpi=150)
    plt.close(fig)
    return funnel, missing


def persona_table(d):
    """Ringkasan per persona; gaji juga dibandingkan dengan median negaranya sendiri."""
    sal = salary(d)
    n_country = sal.groupby(d.Country).transform("count")
    rel = (sal / sal.groupby(d.Country).transform("median")).where(n_country >= MIN_COUNTRY_N)
    g = d.assign(sal=sal, rel=rel).groupby("persona_label")
    t = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(d) * 100,
        "years_code": g.YearsCode.median(),
        "under_25": g.Age.apply(lambda x: x.isin(["Under 18 years old", "18-24 years old"]).mean() * 100),
        "us_share": g.Country.apply(lambda x: (x == "United States of America").mean() * 100),
        "remote": g.RemoteWork.apply(lambda x: (x == "Remote").sum() / x.notna().sum() * 100),
        "job_sat": g.JobSat.mean(),
        "salary": g.sal.median(),
        "vs_country": (g.rel.median() - 1) * 100,
    })
    return t.sort_values("salary", ascending=False)


def salary_chart(t):
    labels = [short(p) for p in t.index]
    y = np.arange(len(t))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)

    a.barh(y, t.salary / 1000, color=BLUE, height=0.6)
    for yi, v in zip(y, t.salary):
        a.text(v / 1000 + 1.5, yi, f"${v / 1000:.0f}k", va="center", color=INK_2)
    a.set_yticks(y, labels)
    a.set_xlim(0, t.salary.max() / 1000 * 1.18)
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}k"))
    a.set_title("Median gaji global (USD/tahun)")
    clean_axes(a)

    colors = [BLUE if v >= 0 else RED for v in t.vs_country]
    b.barh(y, t.vs_country, color=colors, height=0.6)
    for yi, v in zip(y, t.vs_country):
        b.text(v + (1 if v >= 0 else -1), yi, f"{v:+.0f}%", va="center",
               ha="left" if v >= 0 else "right", color=INK_2)
    b.axvline(0, color=INK_2, lw=1)
    lim = max(abs(t.vs_country).max() * 1.35, 10)
    b.set_xlim(-lim, lim)
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
    b.set_title("Selisih dari median gaji negaranya sendiri")
    clean_axes(b)

    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "salary_by_persona.png"), dpi=150)
    plt.close(fig)


def demand_chart(order, labels):
    """Heatmap: % anggota persona yang BELUM memakai skill tsb tapi ingin mempelajarinya."""
    prev = json.load(open(os.path.join(OUT, "artifacts", "cluster_prevalence.json")))
    ids = {v: k for k, v in labels.items()}
    want = np.array([[prev[ids[p]][s]["want"] * 100 for s in DEMAND_SKILLS] for p in order])
    have = np.array([[prev[ids[p]][s]["have"] * 100 for s in DEMAND_SKILLS] for p in order])
    common = have >= 70  # skill yang sudah dipakai mayoritas persona: minat dari sisa kecil, tidak sebanding

    fig, ax = plt.subplots(figsize=(11, 4.6))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(np.ma.masked_where(common, want), cmap=cmap, vmin=0, vmax=want[~common].max(),
              aspect="auto")
    ax.imshow(np.ma.masked_where(~common, np.ones_like(want)),
              cmap=LinearSegmentedColormap.from_list("n", [NEUTRAL, NEUTRAL]), aspect="auto")
    for i in range(len(order)):
        for j in range(len(DEMAND_SKILLS)):
            if common[i, j]:
                ax.text(j, i, f"sudah dipakai\n{have[i, j]:.0f}%", ha="center", va="center",
                        color=INK_2, fontsize=8)
            else:
                dark = want[i, j] > want[~common].max() * 0.55
                ax.text(j, i, f"{want[i, j]:.0f}%", ha="center", va="center",
                        color="white" if dark else INK, fontweight="bold")
    ax.set_xticks(range(len(DEMAND_SKILLS)), [s.split("__", 1)[1] for s in DEMAND_SKILLS])
    ax.set_yticks(range(len(order)), [short(p) for p in order])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(DEMAND_SKILLS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(order)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Persentase yang ingin mempelajari skill ini, di antara yang belum memakainya",
                 pad=28)

    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "skill_demand.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(DOCS, exist_ok=True)
    raw = load_survey()
    funnel, missing = data_quality(raw)
    print("Corong cleaning: " + " -> ".join(f"{n:,}" for _, n in funnel))
    print("Missing rate   : " + ", ".join(f"{c}: {v:.1f}%" for c, v in missing.items()))

    d = pd.read_csv(os.path.join(OUT, "models", "dataset_with_clusters.csv"))
    assert len(d) == funnel[-1][1], f"dataset_with_clusters tidak cocok dengan cleaning: {len(d)}"
    t = persona_table(d)
    salary_chart(t)
    labels = json.load(open(os.path.join(OUT, "artifacts", "persona_labels.json")))
    demand_chart(list(t.index), labels)

    print("\n| Persona | Responden | Median lama coding | Usia < 25 | *Remote* penuh "
          "| Kepuasan kerja (0-10) | Median gaji | Selisih dari median negara |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for p, r in t.iterrows():
        print(f"| {short(p)} | {id_num(r.n)} ({r.share:.0f}%) | {r.years_code:.0f} tahun "
              f"| {r.under_25:.0f}% | {r.remote:.0f}% | {id_num(r.job_sat, 1)} "
              f"| ${r.salary / 1000:.0f}k | {r.vs_country:+.0f}% |")
    med = salary(d).groupby(d.Country).median()
    print(f"\nMedian gaji: global ${salary(d).median() / 1000:.0f}k, "
          f"AS ${med['United States of America'] / 1000:.0f}k, India ${med['India'] / 1000:.0f}k")
    print(f"\nPorsi responden AS per persona: " +
          ", ".join(f"{short(p)} {v:.0f}%" for p, v in t.us_share.items()))
    print(f"Ditulis ke {DOCS}")


if __name__ == "__main__":
    main()
