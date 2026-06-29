# DevPath — Career Profile Segmentation System

Sistem segmentasi profil karir developer berbasis unsupervised machine learning. DevPath mengelompokkan developer ke dalam 6 persona berdasarkan tech stack nyata dari ~23.000 profesional (Stack Overflow Developer Survey 2025), lalu menghasilkan rekomendasi skill gap serta learning roadmap.

---

## Latar Belakang

Mahasiswa dan developer baru seringkali kebingungan dalam meneneutkan karir mereka akbat terlalu banyaknya cabang Computer Science dengan ratusan skills teknologi di masing-masing cabang. Rekomendasi yang tersedia umumnya bersifat generik, tidak berbasis data nyata.

DevPath menjawab pertanyaan: **"Saya cocok ke kelompok developer mana, dan skill apa yang perlu saya pelajari selanjutnya?"** dengan mencocokkan profil skill pengguna terhadap pola nyata dari ~23.000 developer profesional.

---

## Demo


[Demo Video](https://youtu.be/JBCwmD1X12I)


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
      ├─→ Confidence Score        jarak ke centroid cluster vs semua cluster
      ├─→ Radar Chart             coverage skill user per kategori vs centroid cluster
      ├─→ Skill Gap               skill prevalent di cluster yang belum dimiliki user
      └─→ Learning Roadmap        skill gap diranking berdasarkan priority score
```

---

## Dataset

| Atribut         | Detail                                             |
|-----------------|----------------------------------------------------|
| Sumber          | Stack Overflow Developer Survey 2025               |
| Format          | CSV (6 part file)                                  |
| Total Raw       | ~48.867 responden                                  |
| Setelah Cleaning| ~23.387 responden (65.3% retensi)                  |
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
├── app.py                               # Flask backend (versi lokal/Colab)
├── requirements.txt
│
├── templates/
│   └── index.html                       # Frontend SPA
│
├── Outputs/                             
│   ├── artifacts/
│   │   ├── mlb_encoders.pkl             # Fitted MultiLabelBinarizer (6 kolom)
│   │   ├── umap_model.pkl               # Fitted UMAP reducer (Jaccard, 15 dim)
│   │   ├── kmeans_final.pkl             # Fitted K-Means (K=6)
│   │   ├── feature_names.json           # 186 nama fitur binary
│   │   ├── persona_labels.json          # cluster_id untuk nama persona
│   │   └── model_config.json            # Hyperparameter & metrik final
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
