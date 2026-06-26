# Active-Learning ENP Optimization — Core Source Code

Core source code accompanying:
**"Active Learning–Guided Exosome Manufacturing Enables Intravitreal miR-3162 Therapy
for Diabetic Retinopathy"**.

This release provides the **core machine-learning implementation** of the surrogate model
and pairwise ranking validation used in the closed-loop active-learning pipeline
(corresponding to *Algorithm 1* in the Supplementary Information).

---

## Contents

| Path | Description |
|------|-------------|
| `src/probabilistic_booster.py` | Probabilistic gradient-boosting surrogate (PGBM): KRR warm-start, gradient-boosted decision trees with heteroscedastic NLL objective, epistemic + aleatoric uncertainty decomposition. |
| `src/pairwise_ranking.py` | Pairwise-preference ranker: CLR-transformed formulation inputs, desirability scoring, 12-fold parallel cross-validation, confusion matrix + ROC-AUC visualization. Run `python src/pairwise_ranking.py` to reproduce validation figures. |
| `src/loocv_diagnostics.py` | Leave-One-Group-Out cross-validation of the probabilistic surrogate (BOOSTER or GPR). Holds out each unique formulation, collects predicted mean + uncertainty across folds, and reports R²/NLL/MAE with true-vs-predicted diagnostic plots. Run `python src/loocv_diagnostics.py --data sample_data/seed_24.csv`. |
| `src/acquisition.py` | Active-learning acquisition step: trains per-target surrogates, generates a bounds-aware diverse candidate pool, scores via a desirability objective + uncertainty, and selects a UCB-style Exploit/Explore batch (auto-decaying kappa). Run `python src/acquisition.py --data sample_data/seed_24.csv --out next_batch.csv`. |
| `sample_data/seed_24.csv` | Initial seed dataset — 24 ENP formulation experiments (Phase A, R0). Columns: lipid molar ratios (`dc_chol`, `chol`, `dssm`, `dspc`, `dsps`, `dope`) and measured outcomes (`particle_size` nm, `pdi`, `EE` %). Use as demo input for `pairwise_ranking.py`. |

> **Not included** (available from the corresponding author upon reasonable request):
> the full 96-experiment dataset and the round-by-round campaign configuration.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run pairwise ranking cross-validation on the seed dataset
cd code_public_release
python src/pairwise_ranking.py
# → outputs pairwise_cv_results_with_auc.png

# Leave-One-Group-Out diagnostics of the surrogate model
python src/loocv_diagnostics.py --data sample_data/seed_24.csv --model BOOSTER
# → prints R2/NLL/MAE and shows true-vs-predicted plots

# Propose the next batch of experiments (Exploit + Explore)
python src/acquisition.py --data sample_data/seed_24.csv --model BOOSTER --out next_batch.csv
# → writes next_batch.csv
```

Replace `sample_data/seed_24.csv` with your own dataset to use the full pipeline.

---

## Requirements

```
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the accompanying article (ACS Nano, in press).
