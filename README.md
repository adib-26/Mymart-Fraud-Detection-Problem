# MyMart Fraud Detection

An end-to-end, production-minded fraud-triage project for a fictional Malaysian e-commerce retailer. The core deliverable is [`Section A.ipynb`](Section%20A.ipynb), which develops a Multi-Layer Perceptron (MLP) for ranking likely chargebacks into a human-review queue.

The project addresses the classic failure mode of imbalanced fraud detection: a model can report roughly 97% accuracy by predicting every transaction as legitimate while preventing **RM 0** in chargeback losses.

## What Is Improved

- **Future-facing evaluation:** transactions are split chronologically into 60% training, 20% validation, and the latest 20% out-of-time test period.
- **Causal customer features:** history features use only earlier transactions, never future events or labels.
- **Leakage-safe preprocessing:** numeric medians, category vocabulary, scaling statistics, and SMOTENC are all fit on the training period only.
- **Honest model selection:** all candidate strategies are compared with validation PR-AUC; the test period is scored only after selecting the final model.
- **Operational policy analysis:** the notebook compares validation-max-F1, target-recall, and fixed-capacity review policies rather than treating `0.5` as a business decision.
- **Decision-ready reporting:** test PR-AUC includes a bootstrap confidence interval; capacity, lift, review workload, and potential loss prevented are reported in ringgit.

## Repository Structure

```text
Mymart-Fraud-Detection-Problem-main/
├── Section A.ipynb                 # Main modelling notebook
├── mymart_transactions_v1.csv      # Source transaction data (not modified)
├── README.md                       # Project guide
├── Section_B_Diagnosis.pdf         # Pipeline audit report
├── Section-B_output.png            # Section B comparison visual
├── Section_C_BusinessBrief.pdf     # Non-technical business brief
└── Section_D_AILog.pdf             # AI interaction log
```

## Dataset

`mymart_transactions_v1.csv` contains 42,000 transactions from January 2023 to June 2024. The binary target is `chargeback`, where `1` means a confirmed chargeback.

| Field | Role |
|---|---|
| `transaction_id`, `customer_id` | Unique identifiers; excluded from the model |
| `txn_date` | Event time; used for chronological splitting and causal time features |
| `state`, `product_category`, `payment_method`, `device_type` | Categorical transaction context |
| `order_value_myr`, `num_items`, `account_age_days` | Numeric transaction / customer signals |
| `is_new_customer` | Customer-status signal |
| `chargeback` | Target label (`0` legitimate, `1` chargeback) |

The source data is intentionally left unchanged. The notebook validates its schema and fails early for duplicate IDs, invalid labels, invalid timestamps, or invalid monetary / item values.

## Notebook Workflow

1. **EDA and data contract** — class imbalance, missingness, categorical risk, interaction effects, and time stability.
2. **Causal feature engineering** — cyclic time features, order-value features, payment × customer interaction, and prior customer behaviour derived only from earlier events.
3. **Time-aware preparation** — chronological split, train-only imputation / encoding / scaling, and moderate categorical-safe SMOTENC.
4. **MLP comparison** — baseline, class-weighted, SMOTENC, and focal-loss candidates selected by validation PR-AUC only.
5. **Final out-of-time evaluation** — PR-AUC with bootstrap interval, ROC-AUC, confusion matrices, and policy comparison on the untouched latest period.
6. **Model insight and business value** — validation permutation importance, test-period score calibration, review-budget lift, and potential RM loss exposed to intervention.

## Run It

### Local Jupyter

```bash
python -m pip install tensorflow pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter lab
```

Open `Section A.ipynb` and run all cells from top to bottom. Keep `mymart_transactions_v1.csv` in the same directory as the notebook.

### Google Colab

1. Upload `Section A.ipynb` and `mymart_transactions_v1.csv`.
2. Open the notebook in Colab.
3. Run all cells. The first code cell installs only dependencies that are missing.

The notebook uses a fixed seed (`42`) and attempts deterministic TensorFlow operations where the runtime supports them. Small numerical variation across TensorFlow / hardware versions is still normal.

## Key Configuration

Near the top of the notebook, adjust these settings to model MyMart's actual operating conditions:

| Setting | Default | Meaning |
|---|---:|---|
| `RM_PER_CHARGEBACK` | `85` | Direct cost assumed for one missed chargeback |
| `TARGET_RECALL` | `0.90` | Recall target for the high-recall policy scenario |
| `REVIEW_BUDGET_FRACTION` | `0.10` | Share of current transactions routed to manual review |
| `SMOTE_TARGET_RATIO` | `0.50` | Minority / majority ratio after SMOTENC on training data |
| `N_BOOTSTRAPS` | `1000` | Resamples used for the held-out PR-AUC interval |

The top-`k` capacity policy is generally the safest deployment starting point: it has a fixed workload and does not need test labels or a rigid score threshold. Set it from real staffing capacity, not from the default value.

## Metrics That Matter

| Metric | Why it is used |
|---|---|
| PR-AUC | Primary model-selection metric for a rare positive class; its no-skill reference is the chargeback rate |
| Recall | Share of chargebacks sent to intervention |
| Precision / lift | Quality and concentration of the review queue |
| Flag rate / review capacity | Operational workload created by a policy |
| Balanced accuracy | Prevents legitimate-class volume from hiding fraud failure |
| Bootstrap PR-AUC interval | Uncertainty around the final out-of-time ranking estimate |

Accuracy is shown only as a cautionary comparison. It is not used to select the model or operating policy.

## Business Interpretation

The MLP should be treated as a **risk-ranking tool**, not an autonomous decline engine. `Potential RM loss prevented` assumes a flagged true chargeback is successfully stopped or resolved by the downstream review process. It does not include reviewer cost, customer friction, false-positive recovery, or intervention effectiveness; a live pilot must measure those before an ROI claim is made.

Raw scores are not automatically calibrated probabilities, particularly after class weighting, focal loss, or oversampling. If another system needs a probability, fit calibration on a separate recent labelled dataset and evaluate it again on future data.

## Sections B–D

- `Section_B_Diagnosis.pdf` documents the original pipeline audit: leakage, unsuitable accuracy reporting, and missing imbalance handling.
- `Section-B_output.png` visualises the Section B before / after comparison.
- `Section_C_BusinessBrief.pdf` communicates the business case to non-technical stakeholders.
- `Section_D_AILog.pdf` records AI-assisted development interactions for the assignment disclosure.

## Limitations and Next Steps

- The available data has limited behavioural depth, so no result should be interpreted as a safe automatic-blocking policy.
- Historical features assume the transaction stream is complete and available in real time.
- Feature importance is predictive association, not causal explanation; assess fairness, availability, and policy feedback before production use.
- The highest-value next data sources are device fingerprints, payment / address relationships, velocity features, and manual-review dispositions.
- Retrain and re-evaluate when new labels arrive, the schema changes, or monitoring indicates drift.
