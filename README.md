# 🎯 Intelligent Job & Candidate Recommendation System

**A hybrid ML-powered recommendation engine for employment platforms** — it matches job-seekers with job offers, and employers with candidates, using 6 expert scoring criteria refined by a Logistic-Regression model trained on **37,358 real placement records**.

---

## ✨ Overview

This project solves a doubly constrained matching problem found in public employment services:

| Direction | Question it answers |
|---|---|
| 👤 **Candidate mode** | *"Given my CV, which of the registered job offers fit me best?"* |
| 🏢 **Employer mode** | *"Given my job offer, which registered candidate profiles fit best?"* |

The system is **not** a classical collaborative-filtering recommender. It is a **content-based, multi-criteria matching engine**:

- Every candidate ↔ offer pair is scored on **6 expert criteria (C1–C6)** using decision matrices and fuzzy string matching.
- The criterion weights are **automatically re-tuned by a Machine Learning agent** (one Logistic Regression per candidate strategy, trained on real placement data).
- Final rankings are diversified with an **MMR (Maximal Marginal Relevance)** reranker so users get varied, non-redundant results.

---

## 🖼️ Screenshots

### 1 — Main interface (Candidate mode)
Upload a CV (`.txt`, `.pdf`, `.docx`) or fill the form directly. Sidebar controls: number of recommendations, diversity λ, minimum score — with live status of the MongoDB database and the ML model.

![Interface](photo/Interface.png)

### 2 — Automatic CV parsing
The CV is parsed instantly (regex + keyword extraction, no external NLP service): education level, diploma field, years of experience, job title, commune, languages and registration date are detected and used to pre-fill the form.

![After put cv](photo/After%20put%20cv.png)

### 3 — Ranked results
Each match is presented as a card with an **employability score /100**, a classification badge (*Optimale / Bonne / Faible / Nulle*), a natural-language explanation ("Ce candidat correspond car…"), the active strategy, whether the **ML weights were active**, the MMR diversity score and the per-criterion breakdown (expandable).

![show the result](photo/show%20the%20result.png)

### 4 — Diversified Top-N list with criterion details

![More result](photo/More%20result.png)

---

## 🏗️ Architecture

```
        ┌────────────────────────────────────────────────┐
        │      MongoDB Atlas  (employability_db)         │
        │   placements (37,358 real records · referential│
        └───────────────┬────────────────────────────────┘
                        │ pymongo
      ┌─────────────────┴──────────────────┐
      │   ml_agent.py (CLI)                │     dashboard.py (Streamlit)
      │   --train  --demo  --resume        │     👤 Candidate / 🏢 Employer
      └───────────┬────────────────────────┘     └────────────┬───────┘
                  ▼                                           ▼
      ┌──────────────────────────────┐          ┌───────────────────────────┐
      │ agent/features.py            │          │ resume_parser/extractor.py│
      │  record → 20-feature vector  │          │  (.pdf/.docx/.txt → text) │
      │ agent/trainer.py             │          │ resume_parser/parser.py   │
      │  1:1 synthetic negatives     │          │  (regex → structured CV)  │
      │  LogisticRegression per S0–S3│          └─────────────┬─────────────┘
      │  |coef| → weight overrides   │                        ▼
      │  → ml_weight_cache.json      │          ┌───────────────────────────┐
      │  → Mongo referential         │          │ agent/offer_matcher.py    │
      └──────────────────────────────┘          │  aggregate offers / cands │
                  │                             │  score every pair → sort  │
                  ▼                             │  _mmr_rerank (λ slider)   │
      ┌──────────────────────────────┐          └───────────────────────────┘
      │ agent/dynamic_scorer.py      │
      │  ML weights ? base weights   │    shared kernel:
      │  scoring_engine.score_c1–c6  │◄── scoring/scoring_engine.py
      │  TE = Σ wᵢ·sᵢ → /100 → class │    scoring/scoring_config.py
      └──────────────────────────────┘
```

### Component roles

| Directory / file | Role |
|---|---|
| `scoring/scoring_config.py` | Expert knowledge: C1–C6 decision matrices, S0–S3 strategy weights, classification thresholds |
| `scoring/scoring_engine.py` | Stateless scoring kernel — `compute_employability_score()` |
| `agent/features.py` | Record → 20-feature NumPy vector for ML |
| `agent/trainer.py` | Negative sampling, per-strategy LR training, weight blending, persistence |
| `agent/dynamic_scorer.py` | Production scorer: ML weight overrides with automatic fallback to base weights |
| `agent/offer_matcher.py` | Retrieval from MongoDB + MMR diversity reranking |
| `resume_parser/` | File extraction (PDF/DOCX/TXT) + regex-based CV field extraction |
| `ml_agent.py` | CLI entry point: train / demo / score / show weights |
| `dashboard.py` | Streamlit web application (both modes) |
| `nlp.py` | 🧪 Experimental LLM-based CV→JSON parser (Ollama / Mistral) — prototype only |

---

## 🧮 How the scoring works

Six criteria are computed per candidate–offer pair, each in `[0, 1]`:

| Criterion | Meaning | Method |
|---|---|---|
| **C1** | Niveau d'instruction | Fuzzy matrix `M_NI` over 5 education tiers |
| **C2** | Diplômes | `thefuzz` token-set similarity on diploma strings |
| **C3** | Expériences | Years-band matrix **×** job-title proximity |
| **C4** | Langues | Neutral 1.0 (no language data in dataset) |
| **C5** | Ancienneté d'inscription | Banded seniority function (registration priority rule) |
| **C6** | Résidence | 3 mobility scopes: same commune / same wilaya / other |

The final **Employability Score (TE)** is a normalized weighted average:

```
TE = (Σ wᵢ · scoreᵢ) / Σ wᵢ  × 100        →  Optimale ≥75 · Bonne ≥50 · Faible ≥25 · Nulle
```

The weights `wᵢ` depend on the candidate's **strategy** (auto-detected from education level):
**S3** graduate/executive · **S2** mid-level · **S1** execution · **S0** default/uniform.

## 🤖 How the ML refines the weights

```
37,358 real placements  →  1:1 synthetic negatives (shuffled mismatches)
       →  20-feature vectors  →  Logistic Regression (one model per strategy S0–S3)
       →  |coefficients| → per-criterion importances
       →  blend with hand-tuned base weights (confidence-scaled, normalised)
       →  weight overrides  →  ml_weight_cache.json + Mongo `referential`
```

`DynamicScorer` loads these overrides at startup; if none exist it transparently falls back to the expert base weights — so the app always works, with or without an ML model.

Finally, results are reranked with **MMR**: `MMR = λ·relevance − (1−λ)·redundancy`, where λ is user-controlled in the sidebar (1.0 = pure score ordering, 0.0 = maximum diversity).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A MongoDB Atlas account (or local MongoDB) — needed for `--train`, employer mode, and real candidate-mode results
- *(Optional)* an [Ollama](https://ollama.com) server with Mistral — only for `nlp.py`

### 1. Installation

```bash
git clone https://github.com/<your-username>/Recommandation-system.git
cd Recommandation-system

# Base dependencies
pip install -r requirements.txt

# Additional runtime dependencies actually imported by the code
pip install streamlit thefuzz pdfminer.six
# only if you want the experimental nlp.py: pip install ollama pdfplumber
```

### 2. Configuration

Create a `.env` file in the project root (it is gitignored):

```bash
echo "MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/" > .env
```

### 3. Usage

**Run the web app:**
```bash
streamlit run dashboard.py
```

**Train the ML agent on real placements:**
```bash
python ml_agent.py --train          # optional: --limit 5000 for a quick test
```

**Offline demo (no MongoDB required, synthetic data):**
```bash
python ml_agent.py --demo
```

**Score a single resume from the CLI:**
```bash
python ml_agent.py --resume path/to/candidate.txt \
    --offre-ni "Supérieur 1" \
    --offre-diplome "Génie civil" \
    --offre-exp 2 \
    --offre-metier "Ingénieur" \
    --offre-lieu "CHERAGA" \
    --date-offre "2025-06-01"
```

**Inspect current ML weight overrides:**
```bash
python ml_agent.py --show-weights
```

*(Tip: `simple.txt`, `cv_test.pdf` and `cv test.docx` in the repo are sample CVs you can use to test the interface.)*

---

## 📊 ML weight override schema

Stored in `ml_weight_cache.json` and upserted into the Mongo `referential` collection:

```json
{
  "strategy": "S2",
  "trained_at": "2026-04-09T20:52:31",
  "n_samples": 110,
  "accuracy": 1.0,
  "feature_importances": { "c1_ni_match": 0.0, "c2_diplome_match": 0.42, "...": 0 },
  "criterion_importances": { "C1": 0.0, "C2": 2.63, "C3": 2.47, "C4": 1.0, "C5": 0.34, "C6": 0.01 },
  "weight_overrides": { "C1": 0.0268, "C2": 0.3658, "C3": 0.3583, "C4": 0.1473, "C5": 0.0728, "C6": 0.029 }
}
```

## 🔍 CV parser — extracted fields

| Field | Method | Notes |
|---|---|---|
| `demandeur_ni` | Ranked regex keywords (doctorat → master → licence → bac+2 → …) | Conservative default `Moyen` |
| `demandeur_diplome` | Fixed field-of-study keyword list | French/Algerian conventions |
| `demandeur_exp_years` | `"X ans d'expérience"` or sum of `YYYY–YYYY` date ranges | Capped at 40 years |
| `demandeur_metier` | `"Poste:"` lines, then job-title keyword scan | First matching line |
| `demandeur_commune` | `Adresse/Commune:` prefix, then known-commune lookup | Defaults to `ALGER` |
| `date_inscription` | `Date d'inscription:` prefix, 4 date formats | Defaults to today |
| `languages` | Language keyword dictionary (incl. Tamazight) | — |

---

## ⚠️ Known limitations

- All training records are **successful placements** — negatives are synthetically generated, so model accuracy (~99%) must be interpreted with care.
- The LinkedIn-style full market has no failed-application data; the model learns *proximity to past placements*, not true P(hired).
- Geographic scoring is currently limited to Wilaya d'Alger (57 communes).
- CV parsing is regex/keyword-based and tuned to French formatting — Arabic-script CVs are not supported yet (see `nlp.py` prototype).
- `requirements.txt` is incomplete: `streamlit`, `thefuzz` and `pdfminer.six` are imported by the code but must be installed manually (see commands above).

## 🔮 Future work
- Wire the Ollama/LLM parser (`nlp.py`) into the production pipeline for robust multilingual CV extraction.
- Real ranking evaluation (Precision@K, NDCG) on temporally held-out placements.
- Expand geographic coverage to all Algerian wilayas with real distance matrices.
- Fairness audit of the strategy weights and the overqualification penalty in C1.

---

## 📄 License & context

Academic / Master's research project — Recommendation Systems.
Built on anonymised placement data held in MongoDB Atlas.
