import os, json, warnings
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo layout is Outputs/{artifacts,models,templates}; notebook 04 copies them
# next to app.py instead, so allow an override.
OUT_DIR       = os.environ.get("DEVPATH_OUT_DIR", os.path.join(BASE_DIR, "Outputs"))
ARTIFACTS_DIR = os.path.join(OUT_DIR, "artifacts")
MODELS_DIR    = os.path.join(OUT_DIR, "models")

app = Flask(__name__,
            template_folder=os.path.join(OUT_DIR, "templates"),
            static_folder=os.path.join(OUT_DIR, "static"))


def _load(path, loader=None):
    if not os.path.exists(path):
        raise SystemExit(f"Artifact hilang: {path}\n"
                         f"Jalankan dari root repo, atau set DEVPATH_OUT_DIR.")
    if loader:
        return loader(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


print("Loading model artifacts...")
mlb_encoders = _load(os.path.join(ARTIFACTS_DIR, "mlb_encoders.pkl"), joblib.load)
kmeans_model = _load(os.path.join(ARTIFACTS_DIR, "kmeans_final.pkl"), joblib.load)
umap_model   = _load(os.path.join(ARTIFACTS_DIR, "umap_model.pkl"), joblib.load)

feature_names  = _load(os.path.join(ARTIFACTS_DIR, "feature_names.json"))
persona_labels = {int(k): v for k, v in _load(os.path.join(ARTIFACTS_DIR, "persona_labels.json")).items()}
model_config   = _load(os.path.join(ARTIFACTS_DIR, "model_config.json"))
# prevalensi seluruh 186 fitur per cluster -- lihat scripts/build_profiles.py
prevalence     = {int(k): v for k, v in _load(os.path.join(ARTIFACTS_DIR, "cluster_prevalence.json")).items()}
descriptions   = {int(k): v for k, v in _load(os.path.join(ARTIFACTS_DIR, "persona_descriptions.json")).items()}
# stack contoh nyata per cluster, sudah diverifikasi terprediksi balik ke
# clusternya sendiri -- lihat scripts/build_examples.py
examples       = {int(k): v for k, v in _load(os.path.join(ARTIFACTS_DIR, "cluster_examples.json")).items()}
cluster_profiles_raw = _load(os.path.join(MODELS_DIR, "cluster_profiles.json"))
cluster_profiles = {p["cluster_id"]: p for p in cluster_profiles_raw}

COL_ORDER  = ["LanguageHaveWorkedWith","WebframeHaveWorkedWith","DatabaseHaveWorkedWith",
               "PlatformHaveWorkedWith","DevEnvsHaveWorkedWith","AIModelsHaveWorkedWith"]
COL_LABELS = {"LanguageHaveWorkedWith":"Languages","WebframeHaveWorkedWith":"Frameworks & Runtimes",
               "DatabaseHaveWorkedWith":"Databases","PlatformHaveWorkedWith":"Platforms & Tools",
               "DevEnvsHaveWorkedWith":"Dev Environments","AIModelsHaveWorkedWith":"AI Models"}
COL_PREFIX = {"LanguageHaveWorkedWith":"lang","WebframeHaveWorkedWith":"wf",
               "DatabaseHaveWorkedWith":"db","PlatformHaveWorkedWith":"plat",
               "DevEnvsHaveWorkedWith":"env","AIModelsHaveWorkedWith":"ai"}
PREFIX_TO_CAT = {v: COL_LABELS[k] for k, v in COL_PREFIX.items()}

GAP_MIN_PREVALENCE = 0.15   # di bawah ini skill terlalu jarang untuk direkomendasikan

vocabulary = [
    {"name": skill, "col": col, "category": COL_LABELS[col], "prefix": COL_PREFIX[col],
     "key": f"{COL_PREFIX[col]}__{skill}"}
    for col in COL_ORDER
    for skill in mlb_encoders[col].classes_
]


def split_feature(feat):
    prefix, skill = feat.split("__", 1) if "__" in feat else (feat, feat)
    return skill, PREFIX_TO_CAT.get(prefix, "Other")


def encode_binary(user_skills):
    parts, unknown = [], []
    for col in COL_ORDER:
        mlb = mlb_encoders[col]
        skills_in = user_skills.get(col, [])
        known = [s for s in skills_in if s in mlb.classes_]
        unknown.extend([s for s in skills_in if s not in mlb.classes_])
        parts.append(mlb.transform([known]).astype(np.float32))
    return np.hstack(parts), unknown


def recommend(x_binary, cluster_id, top_n=12):
    """Skill yang belum dimiliki user, diberi dua peringkat berbeda.

    skill gap -> seberapa umum skill itu di cluster (have).
    roadmap   -> have + want, dengan want = porsi anggota cluster yang belum
                 memakainya tapi ingin mempelajarinya. Skill yang sedang naik
                 daun di cluster itu (Rust, Go, TypeScript) naik peringkat.
    """
    prev = prevalence.get(cluster_id, {})
    owned = x_binary[0] > 0
    cands = []
    for i, feat in enumerate(feature_names):
        if owned[i]:
            continue
        p = prev.get(feat)
        if not p or p["have"] < GAP_MIN_PREVALENCE:
            continue
        skill, category = split_feature(feat)
        cands.append({"skill": skill, "category": category,
                      "cluster_prevalence": round(p["have"] * 100, 1),
                      "want_ratio": round(p["want"] * 100, 1),
                      "priority_score": round((p["have"] + p["want"]) * 100, 1)})

    return (sorted(cands, key=lambda c: -c["cluster_prevalence"])[:top_n],
            sorted(cands, key=lambda c: -c["priority_score"])[:10])


def category_coverage(x_binary, cluster_id):
    """Per kategori: porsi skill inti cluster yang sudah dimiliki user (untuk radar)."""
    prev = prevalence.get(cluster_id, {})
    out = []
    for col in COL_ORDER:
        core = [i for i, f in enumerate(feature_names)
                if f.startswith(COL_PREFIX[col] + "__")
                and prev.get(f, {}).get("have", 0) >= GAP_MIN_PREVALENCE]
        have = sum(1 for i in core if x_binary[0][i] > 0)
        out.append({"category": COL_LABELS[col], "core": len(core), "have": have,
                    "pct": round(have / len(core) * 100, 1) if core else 0.0})
    return out


@app.route("/")
def index(): return render_template("index.html")


@app.route("/api/vocabulary")
def api_vocabulary(): return jsonify({"skills": vocabulary})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    raw = body.get("skills")
    if not isinstance(raw, dict):
        return jsonify({"error": "Invalid payload."}), 400
    user_skills = {k: v for k, v in raw.items() if k in COL_ORDER and isinstance(v, list)}

    if not user_skills.get("LanguageHaveWorkedWith"):
        return jsonify({"error": "At least one language is required."}), 400
    if sum(len(v) for v in user_skills.values()) < 2:
        return jsonify({"error": "Please add at least 2 skills."}), 400

    x_binary, unknown = encode_binary(user_skills)
    # umap.transform() gagal pada vektor kosong (graph tetangga kosong), dan
    # sebuah profil yang seluruh skill-nya tak dikenal memang tidak bisa dicocokkan.
    if not x_binary.any():
        return jsonify({"error": "None of those skills are recognised. "
                                 "Pick from the suggestions."}), 400
    x_reduced = umap_model.transform(x_binary)

    cluster_id   = int(kmeans_model.predict(x_reduced)[0])
    distances    = kmeans_model.transform(x_reduced)[0]
    sorted_idx   = np.argsort(distances)
    runner_up_id = int(sorted_idx[1])
    # Softmax atas jarak negatif. Rumus lama (1 - d/sum(d)) untuk K=6 terkurung
    # di sekitar 0.7-1.0 sehingga selalu terlihat yakin; ini turun ke ~0.3 saat
    # dua persona teratas berimpit, tapi tetap tinggi kalau sisanya jauh.
    probs        = np.exp(-distances) / np.exp(-distances).sum()
    confidence   = float(probs[cluster_id])

    gap, roadmap = recommend(x_binary, cluster_id)

    return jsonify({
        "cluster_id":            cluster_id,
        "persona_label":         persona_labels.get(cluster_id, f"Cluster {cluster_id}"),
        "confidence":            round(confidence, 3),
        "runner_up_id":          runner_up_id,
        "runner_up_label":       persona_labels.get(runner_up_id, ""),
        "profile":               dict(cluster_profiles.get(cluster_id, {}),
                                      **descriptions.get(cluster_id, {})),
        "skill_gap":             gap,
        "roadmap":               roadmap,
        "active_skills":         [{"name": n, "category": c} for n, c in
                                  (split_feature(f) for i, f in enumerate(feature_names)
                                   if x_binary[0][i] > 0)],
        "category_coverage":     category_coverage(x_binary, cluster_id),
        "total_skills_entered":  sum(len(v) for v in user_skills.values()),
        "unknown_skills":        unknown,
        "all_cluster_distances": [
            {"cluster_id": int(i), "persona_label": persona_labels.get(int(i), f"Cluster {i}"),
             "distance": round(float(distances[i]), 4), "is_match": int(i) == cluster_id}
            for i in sorted_idx],
    })


@app.route("/api/clusters")
def api_clusters():
    return jsonify({"clusters": [dict(p, **descriptions.get(p["cluster_id"], {}),
                                      example_skills=examples.get(p["cluster_id"], {}))
                                 for p in cluster_profiles_raw]})


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "model": model_config.get("best_model"),
                    "k": model_config.get("k"), "n_features": len(feature_names),
                    "reduction": "umap"})


# umap.transform() meng-JIT-compile search function-nya saat pemanggilan pertama
# (~5 detik). Bayar di startup, bukan di request user pertama.
_warm = np.zeros((1, len(feature_names)), dtype=np.float32)
_warm[0, feature_names.index("lang__Python")] = 1
umap_model.transform(_warm)
print(f"  Vocabulary: {len(vocabulary)} skills | Clusters: {len(cluster_profiles)} | Ready.\n")

if __name__ == "__main__":
    app.run(debug=os.environ.get("DEVPATH_DEBUG") == "1",
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5000)))
