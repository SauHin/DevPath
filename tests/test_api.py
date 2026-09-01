"""Smoke test API DevPath.

Jalankan dengan pytest, atau langsung: python tests/test_api.py

Yang dijaga di sini terutama urutan fitur: encode_binary menyusun vektor 186-dim
dengan mengikuti COL_ORDER, dan kalau urutan itu bergeser semua prediksi salah
tanpa error apa pun. Profil kanonik di bawah adalah kanarinya.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as devpath  # noqa: E402  (import lambat: memuat UMAP 77 MB)

client = devpath.app.test_client()


def post(skills):
    return client.post("/api/predict", json={"skills": skills})


def test_health():
    d = client.get("/api/health").get_json()
    assert d["status"] == "ok"
    assert d["k"] == 6
    assert d["n_features"] == 186


def test_vocabulary_carries_column():
    skills = client.get("/api/vocabulary").get_json()["skills"]
    assert len(skills) == 186
    # frontend menurunkan daftar kategorinya dari sini
    assert {"name", "col", "category", "prefix"} <= set(skills[0])


def test_rejects_bad_input():
    assert post({"WebframeHaveWorkedWith": ["React"]}).status_code == 400   # tanpa bahasa
    assert post({"LanguageHaveWorkedWith": ["Python"]}).status_code == 400  # < 2 skill
    assert client.post("/api/predict", json={"skills": "bukan dict"}).status_code == 400
    # seluruh skill tak dikenal -> vektor nol, yang bikin umap.transform() crash
    assert post({"LanguageHaveWorkedWith": ["Cobolang", "Fakescript"]}).status_code == 400


def test_unknown_skills_reported():
    d = post({"LanguageHaveWorkedWith": ["Python", "Fakescript"],
              "DatabaseHaveWorkedWith": ["PostgreSQL"]}).get_json()
    assert d["unknown_skills"] == ["Fakescript"]


def test_canonical_profiles_hit_expected_persona():
    cases = [
        (4, {"LanguageHaveWorkedWith": ["Python", "SQL"], "DatabaseHaveWorkedWith": ["PostgreSQL"]}),
        (1, {"LanguageHaveWorkedWith": ["C#"], "DevEnvsHaveWorkedWith": ["Visual Studio"]}),
        (3, {"LanguageHaveWorkedWith": ["PHP", "JavaScript"], "DatabaseHaveWorkedWith": ["MySQL"]}),
        (0, {"LanguageHaveWorkedWith": ["Java"], "WebframeHaveWorkedWith": ["Spring Boot"]}),
    ]
    for expected, skills in cases:
        d = post(skills).get_json()
        assert d["cluster_id"] == expected, f"{skills} -> {d['persona_label']}"


def test_recommendations_are_not_capped_at_eight():
    # cluster_profiles.json cuma menyimpan 8 skill teratas; rekomendasi sekarang
    # membaca cluster_prevalence.json yang berisi seluruh 186 fitur.
    d = post({"LanguageHaveWorkedWith": ["Python", "SQL"]}).get_json()
    assert len(d["skill_gap"]) == 12
    assert len(d["roadmap"]) == 10


def test_roadmap_ranks_differently_from_gap():
    d = post({"LanguageHaveWorkedWith": ["Python", "SQL"]}).get_json()
    gap = [g["skill"] for g in d["skill_gap"]]
    road = [r["skill"] for r in d["roadmap"]]
    assert gap != road, "roadmap cuma menyalin skill gap lagi"
    assert any(s not in gap for s in road), "roadmap tidak memunculkan skill yang diminati"


def test_confidence_separates_clear_from_ambiguous_profiles():
    clear = post({"LanguageHaveWorkedWith": ["C#"],
                  "DevEnvsHaveWorkedWith": ["Visual Studio"]}).get_json()["confidence"]
    ambiguous = post({"LanguageHaveWorkedWith": ["TypeScript", "JavaScript"],
                      "WebframeHaveWorkedWith": ["React"],
                      "DatabaseHaveWorkedWith": ["PostgreSQL"]}).get_json()["confidence"]
    assert clear > 0.7, clear
    assert ambiguous < 0.5, ambiguous


def test_category_coverage_counts_user_skills():
    cov = post({"LanguageHaveWorkedWith": ["Python", "SQL"],
                "DatabaseHaveWorkedWith": ["PostgreSQL"]}).get_json()["category_coverage"]
    assert len(cov) == 6
    by_cat = {c["category"]: c for c in cov}
    assert by_cat["Languages"]["have"] == 2
    assert by_cat["Databases"]["have"] == 1
    assert by_cat["AI Models"]["have"] == 0


def test_every_persona_has_a_written_description():
    for c in client.get("/api/clusters").get_json()["clusters"]:
        assert len(c.get("description", "")) > 200, c["persona_label"]
        assert len(c.get("responsibilities", [])) >= 5, c["persona_label"]


def test_example_stack_predicts_back_to_its_own_cluster():
    """Kartu "See an example result" harus mendarat di profil yang diklik.

    Contoh yang disusun dari top_skills pernah membuat cluster 2 terprediksi ke
    cluster 5, karena prevalensi mentah didominasi skill yang umum di mana-mana.
    Sekarang contohnya stack responden asli - lihat scripts/build_examples.py.
    """
    for c in client.get("/api/clusters").get_json()["clusters"]:
        example = c.get("example_skills")
        assert example, c["persona_label"]
        d = post(example).get_json()
        assert d["cluster_id"] == c["cluster_id"], (
            f'{c["persona_label"]} -> {d["persona_label"]}')


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} lulus")
    sys.exit(1 if failed else 0)
