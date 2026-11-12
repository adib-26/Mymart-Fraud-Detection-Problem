# MyMart Fraud Detection


A multi-section deep learning assignment built around a fictional Malaysian e-commerce retailer, **MyMart**, facing a chargeback fraud problem. The goal across all four sections is to design, audit, and communicate a production-ready fraud detection pipeline using a Multi-Layer Perceptron (MLP).

---

## Project Structure

```
Mymart Fraud Detection Problem
├── Section A.ipynb       # Main modelling notebook (EDA → MLP → Evaluation)
├── Section-B_output.png           # Section B: before/after metric comparison chart
├── Section_B_Diagnosis.pdf        # Section B: written audit report
├── Section_C_BusinessBrief.pdf    # Section C: non-technical business brief
└── Section_D_AILog.pdf            # Section D: AI interaction log
```

---

## The Problem

MyMart's operations team inherited a fraud detection script from a previous consultant. The model reported **97% accuracy** in testing but flagged **almost zero chargebacks** in production — a classic symptom of a broken ML pipeline on an imbalanced dataset.

The dataset (`mymart_transactions_v1.csv`) contains ~42,000 transactions with a severe **96.9% / 3.1%** legitimate-to-chargeback split. Each missed chargeback costs MyMart **RM 85** in direct losses and penalties.

---

## Section A — Model Development (`Section A.ipynb`)

An end-to-end pipeline built in Google Colab, structured as follows:

### A1. Data Loading & EDA
- Inspects class imbalance, missing values, and feature distributions
- Visualises the "needle in a haystack" problem: 3.1% fraud across 42,000 rows

### A2. Data Preparation
- Median imputation for missing `account_age_days`
- One-hot encoding for categorical features (`state`, `product_category`, `payment_method`, `device_type`)
- Leakage-safe customer history features: prior transaction count, time since the previous transaction, and prior average order value
- **Stratified train/validation/test split before scaling** to prevent data leakage
- `StandardScaler` fitted on training data only; applied (not re-fitted) on val/test

### A3. Model Architecture — The "Financial Shield"

An MLP chosen deliberately over simpler models for its ability to capture **non-linear interactions** between features (e.g., high-value order + new account + specific device type):

```
Input → Dense(128) → BatchNorm → LeakyReLU(0.1) → Dropout(0.3)
      → Dense(64)  → BatchNorm → LeakyReLU(0.1) → Dropout(0.2)
      → Dense(1, sigmoid)
```

**Training strategies for imbalance:**
- `class_weight='balanced'` to penalise missed chargebacks more heavily
- SMOTENC categorical-safe oversampling on the training set
- Focal Loss experiment to force attention on hard cases
- `ReduceLROnPlateau` + `EarlyStopping` on validation PR-AUC
- Repeated stratified training-fold stability checks with fold-specific scaling

**Metrics used:** Precision-Recall AUC, Recall, F1-Score (not accuracy — a model predicting all-legitimate scores 97% accuracy trivially)

### A4. Evaluation & Threshold Tuning
- Precision-Recall trade-off curve generated to find the optimal threshold
- Precision-Recall trade-off curve generated to compare operating points
- Balanced threshold selected by maximum validation F1
- High-recall threshold selected as the **highest (strictest)** validation threshold that meets the 90% recall target; if the target is unattainable, the notebook reports and uses the best available recall point
- Permutation importance is measured on validation data so the held-out test set remains reserved for final evaluation
- Probability calibration plot assesses whether scores are reliable probabilities or should be used only for ranking

### A5. Results

| Metric | Naive Baseline | Optimised MLP |
|---|---|---|
| Accuracy | ~97% | Reported from the selected threshold |
| Recall (Chargebacks) | 0.00% | Validation-tuned; reported on test |
| F1-Score (Class 1) | 0.0000 | Reported on test |
| Chargebacks Detected | 0 | Computed from the test split |
| Potential Loss Prevented | RM 0 | Computed as detected chargebacks × RM 85 |

Results are recomputed when the notebook runs; the test split is not used to select the model or thresholds.

---

## Section B — Pipeline Audit

Three critical bugs were identified in the original consultant script and fixed:

| # | Bug | Root Cause | Business Impact |
|---|---|---|---|
| 1 | **Data Leakage** | `StandardScaler` fitted on the full dataset before the train/test split | Model "sees" future data during training; production performance collapses |
| 2 | **Wrong Metric** | `accuracy` used on a 97/3 imbalanced dataset | A model predicting all-legitimate achieves 97% accuracy; fraud goes undetected |
| 3 | **No Imbalance Handling** | No `class_weight` or resampling applied | Model ignores the minority class; 0% recall on actual chargebacks |

The audited pipeline corrects all three: split-before-scale, class weights, stratified split, and Recall as the primary metric. See `Section_B_Diagnosis.pdf` and `Section-B_output.png` for the written diagnosis and comparison.

---

## Section C — Business Brief (`Section_C_BusinessBrief.pdf`)

A non-technical brief for MyMart's operations and finance stakeholders covering:
- Why 97% accuracy is misleading in this context
- What the revised model actually detects (and misses)
- ROI framing using the RM 85-per-chargeback cost figure
- Deployment recommendation: high-recall model with manual review queue for flagged transactions

---

## Section D — AI Interaction Log (`Section_D_AILog.pdf`)

Documents AI-assisted development interactions in accordance with assignment disclosure requirements.

---

## Dependencies

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | MLP model, Focal Loss, callbacks |
| `scikit-learn` | Preprocessing, metrics, permutation importance, calibration |
| `imbalanced-learn` | SMOTENC categorical-safe oversampling |
| `pandas` / `numpy` | Data wrangling |
| `matplotlib` / `seaborn` | Visualisations |

---

## Running the Notebook

1. Place `mymart_transactions_v1.csv` beside the notebook, or upload it to `/content/` in Google Colab
2. Open `Section A.ipynb` in Colab
3. Run all cells in order — dependencies are installed in Cell 2 when missing
4. Review the validation-tuned thresholds and recomputed test-set business-impact table

> Section B is represented by the diagnosis report and comparison chart included in this project.

---

## Key Takeaways

- **Accuracy is a misleading metric for imbalanced fraud data.** Always evaluate with Precision, Recall, F1, and AUC-PR.
- **Data leakage is silent.** Fitting a scaler before splitting inflates validation scores and hides production failures.
- **Threshold tuning matters.** The 0.5 default is not optimal for minority-class detection; calibrate it against the actual business cost of false negatives vs. false positives.
- **Keep the test set honest.** Select the winning strategy, thresholds, and feature explanations without using test labels; use test data only for final assessment and business estimates.
