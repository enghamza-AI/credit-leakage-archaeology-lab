# 🏦 Leakage Archaeology Lab

<div align="center">

**Excavating the 5 sins of credit ML — and building a detector that catches them automatically.**

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Leakr%20Live%20Demo-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/enghamza-AI/Leakr)
[![GitHub](https://img.shields.io/badge/GitHub-credit--leakage--archaeology--lab-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/enghamza-AI/credit-leakage-archaeology-lab)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## 🎯 What Is This?

A model scored **AUC 0.94** on a credit default prediction task. Everyone celebrated. The model was deployed.

Six months later — it was making terrible predictions in production. Loans it called safe were defaulting. The bank lost millions.

**The model had cheated during training. It didn't learn patterns. It learned to peek at the answer.**

This project identifies, demonstrates, and measures the 5 most dangerous forms of data leakage in ML pipelines — using the real Home Credit Default Risk dataset (350k loans, 120+ features). It then ships a reusable **auto-leakage detector** that can scan any tabular dataset and flag these sins before training begins.

> An auto-leakage detector is a real MLOps tool that teams use in production. This project builds one from scratch.

---

## 🔴 The 5 Leakage Sins

| Sin | Name | What Goes Wrong | AUC Impact |
|-----|------|----------------|------------|
| 1 | **Target Encoding Leak** | Encode categories using full dataset stats before splitting | Inflated AUC |
| 2 | **Feature From Target** | A column derived from post-outcome data stays in features | Inflated AUC |
| 3 | **Timestamp Scaling** | Scaler fitted on all data — future stats contaminate training | Inflated AUC |
| 4 | **Group Overlap** | Same borrower in both train and test after random split | Inflated AUC |
| 5 | **Duplicate ID Leak** | ETL bug creates identical rows split across train and test | Inflated AUC |

For each sin, the project measures:
```
Leaky AUC  :  0.91  ← what the sin makes you THINK you have
Honest AUC :  0.71  ← what you'd actually get in production
Inflation  : +0.20  ← the lie
```

---

## 🏗️ Project Structure

```
credit-leakage-archaeology-lab/
│
├── load_data.py              ← Load CSV + sanity check (shape, missing, duplicates)
├── eda.py                    ← Explore data + hunt suspicious correlations
├── clean.py                  ← Safe cleaning — no stat-based transforms (those leak)
├── split.py                  ← Random split, stratified split, Group K-Fold
├── train.py                  ← Honest Pipeline baseline — establishes real AUC
│
├── sin_01_target_encoding.py ← Demonstrate + measure Sin 1
├── sin_02_feature_from_target.py
├── sin_03_timestamp_scaling.py
├── sin_04_group_overlap.py
├── sin_05_duplicate_id.py
│
├── leakage_detector.py       ← Auto-detector class — scans any dataset for all 5 sins
├── visualize.py              ← All charts: AUC explainer, comparison, radar, per-sin detail
├── app.py                    ← Streamlit dashboard — 5 pages, dark theme
│
├── data/
│   └── application_train.csv ← Download from Kaggle (link below)
├── outputs/                  ← Auto-generated charts saved here
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/enghamza-AI/credit-leakage-archaeology-lab.git
cd credit-leakage-archaeology-lab
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Go to → [Home Credit Default Risk on Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

Download **`application_train.csv`** and place it in the `data/` folder.

### 4. Run in order
```bash
python load_data.py          # Day 1 — load + sanity check
python eda.py                # Day 1 — explore + hunt leakage signals
python clean.py              # Day 1 — safe cleaning
python split.py              # Day 2 — splitting strategies
python train.py              # Day 2 — establish honest baseline AUC

python sin_01_target_encoding.py      # Day 3
python sin_02_feature_from_target.py  # Day 3
python sin_03_timestamp_scaling.py    # Day 4
python sin_04_group_overlap.py        # Day 4
python sin_05_duplicate_id.py         # Day 5

python leakage_detector.py   # Day 5 — run the auto-detector
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📊 The Auto-Leakage Detector

The crown jewel of this project. A reusable class that scans any tabular dataset for all 5 leakage archetypes before training begins.

```python
from leakage_detector import LeakageDetector

detector = LeakageDetector(df, target_col='TARGET', id_col='SK_ID_CURR')
report   = detector.run_full_scan()
detector.print_report()
```

**Output:**
```
══════════════════════════════════════════════════════════════
  LEAKAGE DETECTOR — SCAN REPORT
══════════════════════════════════════════════════════════════

  Sin 1: Target Encoding
  Risk   : 🟡 MEDIUM
  Finding: 16 categorical columns at risk of target encoding leak.
  Fix    : Encode AFTER split or use Pipeline with TargetEncoder.

  Sin 2: Feature From Target
  Risk   : ✅ CLEAN
  Finding: No features show suspicious correlation with target.

  Sin 5: Duplicate Rows
  Risk   : ✅ CLEAN
  Finding: 0 exact duplicate rows.

══════════════════════════════════════════════════════════════
  OVERALL SCORE : 8/100
  OVERALL RISK  : 🟢 LOW
══════════════════════════════════════════════════════════════
```

The detector returns:
- **Risk level** per sin: CLEAN / LOW / MEDIUM / HIGH / CRITICAL
- **Specific columns** that triggered each warning
- **Overall leakage risk score** (0–100)
- **Recommended fix** for each detected issue

---

## 🧠 Concepts You Will Understand After This Project

| Concept | Where It Appears |
|---------|-----------------|
| Data leakage — what it is and why it destroys production models | Every file |
| AUC — what it measures, why it can lie, how to compute it by hand | `train.py`, `visualize.py` |
| Target encoding — what it is and when it becomes a sin | `sin_01` |
| sklearn Pipeline — why it prevents scaling leakage | `train.py`, `sin_03` |
| Group K-Fold cross-validation — when random split is wrong | `split.py`, `sin_04` |
| Feature correlation with target — how to detect derived features | `eda.py`, `leakage_detector.py` |
| Stratified split — preserving class balance | `split.py` |
| Duplicate detection — one line that saves a model | `clean.py`, `sin_05` |
| StandardScaler + SimpleImputer inside Pipelines | `train.py` |
| Streamlit dashboard deployment | `app.py` |

---

## 📈 The Core Pattern — Repeated Across All 5 Sins

Every sin file follows the same structure:

```
1. Prepare data
2. LEAKY approach   → train model → record inflated AUC
3. HONEST approach  → train model → record real AUC
4. Compare          → measure inflation = the lie
5. Detect           → show how to catch this in any future dataset
6. Save chart       → PNG saved to outputs/
```

This structure makes each sin independently runnable and directly comparable.

---

## 🔬 The Pipeline Rule — Most Important Concept

```python
# ❌ WRONG — scaler sees test data
scaler.fit(X_all)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ✅ CORRECT — Pipeline enforces train-only fitting
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression())
])
pipeline.fit(X_train, y_train)      # scaler learns from train only
pipeline.predict_proba(X_test)      # applies train stats to test
```

Every transformation that uses statistics (mean, std, default rate) must happen **inside a Pipeline, after the split.** This is not optional. It is the difference between a production-ready model and one that quietly lies about its own performance.

---

## 🗂️ Dataset

**Home Credit Default Risk**
- Source: [Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- File needed: `application_train.csv`
- Size: ~160MB, 307,511 rows, 122 columns
- Target: `TARGET` — 0 = loan repaid, 1 = defaulted
- Default rate: ~8.07% (imbalanced — this is why we use AUC not accuracy)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading, manipulation, groupby operations |
| `numpy` | Numerical operations, array manipulation |
| `scikit-learn` | Pipelines, models, splitters, metrics |
| `matplotlib` | All chart generation |
| `streamlit` | Interactive dashboard |
| `LogisticRegression` | Baseline model (same across all sins for fair comparison) |

---

## 🤗 Live Demo

Try the auto-detector on any dataset without running any code:

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Try%20Leakr%20Live-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/enghamza-AI/Leakr)

Upload your CSV → select your target column → get an instant leakage risk report.

---

## 👤 Author

**Hamza** — BSAI Student, China | Self-studying AI Systems Engineering

Building a 55-project portfolio targeting frontier AI labs and YC-backed startups.

[![GitHub](https://img.shields.io/badge/GitHub-enghamza--AI-181717?style=flat-square&logo=github)](https://github.com/enghamza-AI)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-enghamza--AI-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/enghamza-AI)

---

## 📌 Part of the Diamond AI Roadmap

This is **Stage 1 · Week 4** of an 11-stage, 55+ project AI systems engineering roadmap.

| Stage | Theme |
|-------|-------|
| ✅ Stage 1 | Signal Intelligence & Failure Forensics |
| Stage 2 | Decision Intelligence & Metric Engineering |
| ... | ... |
| Stage 11 | Production AI Systems — Serving, Monitoring & Reliability |

---

