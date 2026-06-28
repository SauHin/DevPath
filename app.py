import traceback
import os, json, warnings
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")
app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

print("Loading model artifacts...")
mlb_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "mlb_encoders.pkl"))
kmeans_model = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans_final.pkl"))

umap_model, svd_model = None, None
umap_path = os.path.join(ARTIFACTS_DIR, "umap_model.pkl")
svd_path  = os.path.join(ARTIFACTS_DIR, "svd_model.pkl")
if os.path.exists(umap_path):
    umap_model = joblib.load(umap_path)
    REDUCTION_MODE = "umap"
    print("  Reduction: UMAP")
elif os.path.exists(svd_path):
    svd_model = joblib.load(svd_path)
    REDUCTION_MODE = "svd"
    print("  Reduction: SVD")
else:
    raise FileNotFoundError("No reduction model found")

with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"))  as f: feature_names  = json.load(f)
with open(os.path.join(ARTIFACTS_DIR, "persona_labels.json")) as f: persona_labels = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(ARTIFACTS_DIR, "model_config.json"))   as f: model_config   = json.load(f)
with open(os.path.join(MODELS_DIR,    "cluster_profiles.json")) as f: cluster_profiles_raw = json.load(f)
cluster_profiles = {p["cluster_id"]: p for p in cluster_profiles_raw}

COL_ORDER  = ["LanguageHaveWorkedWith","WebframeHaveWorkedWith","DatabaseHaveWorkedWith",
               "PlatformHaveWorkedWith","DevEnvsHaveWorkedWith","AIModelsHaveWorkedWith"]
COL_LABELS = {"LanguageHaveWorkedWith":"Languages","WebframeHaveWorkedWith":"Frameworks & Runtimes",
               "DatabaseHaveWorkedWith":"Databases","PlatformHaveWorkedWith":"Platforms & Tools",
               "DevEnvsHaveWorkedWith":"Dev Environments","AIModelsHaveWorkedWith":"AI Models"}
COL_PREFIX = {"LanguageHaveWorkedWith":"lang","WebframeHaveWorkedWith":"wf",
               "DatabaseHaveWorkedWith":"db","PlatformHaveWorkedWith":"plat",
               "DevEnvsHaveWorkedWith":"env","AIModelsHaveWorkedWith":"ai"}

vocabulary = []
for col in COL_ORDER:
    if col not in mlb_encoders: continue
    for skill in mlb_encoders[col].classes_:
        vocabulary.append({"name": skill, "category": COL_LABELS[col],
                           "prefix": COL_PREFIX[col], "key": f"{COL_PREFIX[col]}__{skill}"})

print(f"  Vocabulary: {len(vocabulary)} skills | Clusters: {len(cluster_profiles)} | Ready.\n")

def encode_user(user_skills):
    parts, unknown = [], []
    for col in COL_ORDER:
        if col not in mlb_encoders:
            parts.append(np.zeros((1, 1), dtype=np.float32)); continue
        mlb = mlb_encoders[col]
        skills_in = user_skills.get(col, [])
        known  = [s for s in skills_in if s in mlb.classes_]
        unknown.extend([s for s in skills_in if s not in mlb.classes_])
        parts.append(mlb.transform([known]).astype(np.float32))
    x_binary = np.hstack(parts)
    if REDUCTION_MODE == "umap":
        x_reduced = umap_model.transform(x_binary)
    else:
        x_reduced = svd_model.transform(csr_matrix(x_binary))
        n80 = int(np.searchsorted(np.cumsum(svd_model.explained_variance_ratio_)*100, 80)) + 1
        x_reduced = x_reduced[:, :n80]
    return x_reduced, x_binary, unknown

def compute_skill_gap(x_binary, cluster_id, top_n=12):
    profile = cluster_profiles.get(cluster_id, {})
    skill_prev = {s["feature"].split("__",1)[1] if "__" in s["feature"] else s["feature"]:
                  float(s["prevalence_pct"])/100
                  for s in profile.get("top_skills", [])}
    prefix_to_cat = {v: COL_LABELS[k] for k, v in COL_PREFIX.items()}
    gaps = []
    for i, feat in enumerate(feature_names):
        if x_binary[0][i] > 0: continue
        prefix, skill = feat.split("__",1) if "__" in feat else (feat, feat)
        prev = skill_prev.get(skill, 0.0)
        if prev < 0.15: continue
        gaps.append({"skill": skill, "category": prefix_to_cat.get(prefix,"Other"),
                     "cluster_prevalence": round(prev*100, 1), "gap_score": round(prev*100,1)})
    return sorted(gaps, key=lambda x: -x["gap_score"])[:top_n]

def compute_roadmap(x_binary, cluster_id, top_n=10):
    gaps = compute_skill_gap(x_binary, cluster_id, top_n=50)
    roadmap = [dict(g, want_ratio=0.6, priority_score=round(g["gap_score"]*1.6, 2)) for g in gaps]
    return sorted(roadmap, key=lambda x: -x["priority_score"])[:top_n]

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/vocabulary")
def api_vocabulary(): return jsonify({"skills": vocabulary})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body        = request.get_json(force=True)
    user_skills = body.get('skills', {})

    if not user_skills.get('LanguageHaveWorkedWith'):
        return jsonify({'error': 'At least one language is required.'}), 400
    if sum(len(v) for v in user_skills.values()) < 2:
        return jsonify({'error': 'Please add at least 2 skills.'}), 400

    try:
        x_reduced, x_binary, unknown = encode_user(user_skills)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Encoding error: {str(e)}'}), 500

    try:
        cluster_id   = int(kmeans_model.predict(x_reduced)[0])
        persona_name = persona_labels.get(cluster_id, f'Cluster {cluster_id}')
        profile      = dict(cluster_profiles.get(cluster_id, {}))
        distances    = kmeans_model.transform(x_reduced)[0]
        sorted_idx   = np.argsort(distances)
        confidence   = float(1 - distances[cluster_id] / distances.sum())
        runner_up_id = int(sorted_idx[1]) if len(sorted_idx) > 1 else None
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


    try:
        # ── Skill gap & roadmap ────────────────────────────────
        skill_gap = compute_skill_gap(x_binary, cluster_id)
        roadmap   = compute_roadmap(x_binary, cluster_id)

        prefix_to_cat = {v: COL_LABELS[k] for k, v in COL_PREFIX.items()}
        active_skills = [
            {'name': f.split('__',1)[1] if '__' in f else f,
             'category': prefix_to_cat.get(f.split('__')[0], 'Other')}
            for i, f in enumerate(feature_names) if x_binary[0][i] > 0
        ]

        all_cluster_distances = [
            {'cluster_id':    int(i),
             'persona_label': persona_labels.get(int(i), f'Cluster {i}'),
             'distance':      round(float(distances[i]), 4),
             'is_match':      int(i) == cluster_id}
            for i in sorted_idx
        ]
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Post-processing error: {str(e)}'}), 500

    return jsonify({
        'cluster_id':             cluster_id,
        'persona_label':          persona_name,
        'confidence':             round(confidence, 3),
        'runner_up_id':           runner_up_id,
        'runner_up_label':        persona_labels.get(runner_up_id, '') if runner_up_id is not None else '',
        'profile':                profile,
        'skill_gap':              skill_gap,
        'roadmap':                roadmap,
        'active_skills':          active_skills,
        'total_skills_entered':   sum(len(v) for v in user_skills.values()),
        'unknown_skills':         unknown,
        'all_cluster_distances':  all_cluster_distances,
    })

@app.route("/api/clusters")
def api_clusters(): return jsonify({"clusters": cluster_profiles_raw})

@app.route("/api/health")
def api_health():
    return jsonify({"status":"ok","model":model_config.get("best_model"),"k":model_config.get("k"),
                    "n_features":len(feature_names),"reduction":REDUCTION_MODE})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)