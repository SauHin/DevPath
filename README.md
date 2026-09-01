# DevPath — Career Profile Segmentation System

Sistem segmentasi profil karir developer berbasis unsupervised machine learning. DevPath mengelompokkan developer ke dalam 6 persona berdasarkan tech stack nyata dari ~23.000 profesional (Stack Overflow Developer Survey 2025), lalu menghasilkan rekomendasi skill gap serta learning roadmap.

---

## Latar Belakang

Mahasiswa dan developer baru seringkali kebingungan dalam meneneutkan karir mereka akbat terlalu banyaknya cabang Computer Science dengan ratusan skills teknologi di masing-masing cabang. Rekomendasi yang tersedia umumnya bersifat generik, tidak berbasis data nyata.

DevPath menjawab pertanyaan: **"Saya cocok ke kelompok developer mana, dan skill apa yang perlu saya pelajari selanjutnya?"** dengan mencocokkan profil skill pengguna terhadap pola nyata dari ~23.000 developer profesional.

---

## Demo


Youtube link: [Demo Video](https://youtu.be/JBCwmD1X12I)

> Catatan: video ini masih merekam antarmuka versi lama. Tampilan saat ini
> mengikuti `DESIGN.md`.


> Tersedia sebagai web app lokal (Flask) atau via Cloudflare Tunnel di Google Colab.

---

## Cara Kerja Sistem

```
User Input (skill)
      │
      ▼
Multi-hot Binary Encoding        → 186-dim binary vector
      │                            (42 lang + 28 wf + 30 db + 42 plat + 27 env + 17 ai)
      ▼
UMAP Transform (Jaccard)         → 15-dim dense embedding
      │                            (menggunakan model yang sudah di-fit dari training data)
      ▼
K-Means Predict (K=6)            → cluster_id (0–5)
      │
      ├─→ Persona Label           "Python Backend & Data Developer"
      ├─→ Confidence Score        softmax atas jarak ke seluruh centroid
      ├─→ Trait Coverage          cakupan skill inti cluster per kategori
      ├─→ Skill Gap               skill prevalent di cluster yang belum dimiliki user
      └─→ Learning Roadmap        gap diranking ulang dengan sinyal WantToWorkWith
```

---

## Menjalankan Secara Lokal

```bash
git clone https://github.com/SauHin/DevPath.git && cd DevPath
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000`. Startup butuh sekitar 30 detik pada run pertama: load
UMAP 77 MB lalu memanggilnya sekali untuk trigger JIT numba. Setelah itu
tiap prediksi berjalan di bawah 10 ms.

Cek status API: `curl localhost:5000/api/health`

| Variabel        | Default             | Guna                                       |
|-----------------|---------------------|--------------------------------------------|
| `PORT`          | `5000`              | Port HTTP                                  |
| `HOST`          | `127.0.0.1`         | Set `0.0.0.0` untuk diakses dari luar       |
| `DEVPATH_DEBUG` | off                 | `1` untuk mode debug Flask                  |
| `DEVPATH_OUT_DIR` | `./Outputs`       | Lokasi `artifacts/`, `models/`, `templates/`|

Test: `python tests/test_api.py` (atau `pytest tests/`).

---

## Deploy

`Dockerfile` yang tersedia menargetkan **Hugging Face Spaces** (SDK: Docker, port
7860). Push hanya file yang dibutuhkan ke repo Space — jangan mirror repo GitHub
ini, karena history-nya membawa ~140 MB CSV mentah.

Footprint runtime setelah model dimuat adalah ~427 MB RSS, jadi free tier dengan
batas 512 MB (mis. Render) terlalu mepet; Hugging Face Spaces (16 GB) aman.

> Catatan teknis: sempat diuji mengganti UMAP saat inference dengan voting k-NN
> Jaccard langsung di ruang biner, supaya artifact turun dari 77 MB ke ~300 KB.
> Validasi leave-one-out (`scripts/build_knn_index.py`) hanya mencapai **93,45%**
> kesepakatan dengan label asli, di bawah ambang 95% yang ditetapkan, jadi UMAP
> dipertahankan.

---

## Dataset

| Atribut         | Detail                                             |
|-----------------|----------------------------------------------------|
| Sumber          | Stack Overflow Developer Survey 2025               |
| Format          | CSV (6 part file)                                  |
| Total Raw       | ~48.867 responden                                  |
| Setelah Cleaning| ~23.387 responden (47.9% retensi)                  |
| Kolom           | 172 kolom per part                                 |

Sumber: [Stack Overflow Annual Developer Survey](https://survey.stackoverflow.co/)

---

## Struktur Project

```
DevPath/
│
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis
│   ├── 02_FeatureExtraction_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb                # Training, evaluasi, simpan model
│   └── 04_ApplicationLayer.ipynb        # Generate app files + deploy
│
├── app.py                               # Flask backend
├── requirements.txt
├── Dockerfile                           # target Hugging Face Spaces
├── PRODUCT.md                           # kebenaran produk (audiens, batasan, bukti)
├── DESIGN.md                            # sistem desain, ditulis dari build
│
├── scripts/
│   ├── build_profiles.py                # prevalensi 186 skill + sinyal WantToWorkWith
│   ├── build_examples.py                # stack contoh nyata per cluster (kartu landing)
│   └── build_knn_index.py               # eksperimen pengganti UMAP (lihat catatan Deploy)
│
├── tests/
│   └── test_api.py
│
├── Outputs/                             
│   ├── artifacts/
│   │   ├── mlb_encoders.pkl             # Fitted MultiLabelBinarizer (6 kolom)
│   │   ├── umap_model.pkl               # Fitted UMAP reducer (Jaccard, 15 dim)
│   │   ├── kmeans_final.pkl             # Fitted K-Means (K=6)
│   │   ├── feature_names.json           # 186 nama fitur binary
│   │   ├── persona_labels.json          # cluster_id untuk nama persona
│   │   ├── cluster_prevalence.json      # prevalensi & minat 186 skill per cluster
│   │   ├── persona_descriptions.json    # deskripsi persona, di-key cluster_id
│   │   └── model_config.json            # Hyperparameter & metrik final
│   │
│   ├── templates/
│   │   └── index.html                   # Frontend: satu file, tanpa build step
│   │
│   ├── static/fonts/                    # dua typeface woff2 self-hosted
│   │
│   └── models/
│       ├── cluster_profiles.json        # Profil lengkap 6 cluster
│       ├── cluster_assignments.csv      # Label cluster per respondent
│       └── dataset_with_clusters.csv    # Dataset  dengan cluster label
│
└── Dataset/                          
    ├── part_1_SO_survey.csv
    ├── part_2_SO_survey.csv
    ├── part_3_SO_survey.csv
    ├── part_4_SO_survey.csv
    ├── part_5_SO_survey.csv
    └── part_6_SO_survey.csv
```
