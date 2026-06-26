"""
ExoAI — Leave-One-Out Cross-Validation (LOO-CV) Diagnostics
============================================================

Standalone backbone (no GUI) for evaluating the probabilistic surrogate model
used in the ExoAI lipid-nanoparticle optimization pipeline.

For every unique formulation (a "group" of replicate rows) the model is trained
on all other groups and asked to predict the held-out one. Predicted mean and
uncertainty are collected across all folds and scored with R2, NLL and MAE for
each target (particle Size, PDI, EE).

Usage
-----
    python loocv_diagnostics.py --data my_data.csv --model BOOSTER
    python loocv_diagnostics.py --data my_data.csv --model GPR --jobs 4 --no-plot

CSV requirements
----------------
Columns: dc_chol, chol, dssm, dspc, dsps, dope, NP, particle_size, pdi, EE
(Each row is one measurement; identical feature rows are treated as one group.)
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler

# Optional deep-ensemble booster (the default ExoAI model).
try:
    from probabilistic_booster_v3 import MultiOutputDeepEnsemble
except ImportError:
    MultiOutputDeepEnsemble = None

# ----------------------------------------------------------------------
# Feature definition
# ----------------------------------------------------------------------
LIPID_KEYS = ['dc_chol', 'chol', 'dssm', 'dspc', 'dsps', 'dope']
FEATURES = LIPID_KEYS + ['NP']            # NP = N/P ratio (non-compositional)


# ----------------------------------------------------------------------
# Pre-processing helpers
# ----------------------------------------------------------------------
def clr_transform(X):
    """Centered Log-Ratio transform for compositional (lipid %) data."""
    X_mat = np.array(X, dtype=float)
    log_x = np.log(X_mat + 1e-8)
    gm = np.mean(log_x, axis=1, keepdims=True)
    return log_x - gm


def load_data(filepath):
    """Load CSV and assign a Group_ID to each unique formulation."""
    df = pd.read_csv(filepath)
    missing = [k for k in FEATURES if k not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    input_str = df[FEATURES].astype(str).agg('-'.join, axis=1)
    df['Group_ID'] = pd.factorize(input_str)[0]

    return (df[FEATURES].values,
            df['particle_size'].values,
            df['pdi'].values,
            df['EE'].values,
            df['Group_ID'].values)


# ----------------------------------------------------------------------
# Model training + prediction
# ----------------------------------------------------------------------
def train_and_predict(model_type, X_train, y_train, X_candidates, groups=None):
    """
    Train the chosen surrogate and return (mean, std) predictions on X_candidates.

    CLR is applied only to the compositional lipid columns; the N/P ratio is
    min-max scaled separately so it cannot dominate the distance metric.
    """
    num_lipids = len(LIPID_KEYS)

    X_train_clr = clr_transform(X_train[:, :num_lipids])
    X_cand_clr = clr_transform(X_candidates[:, :num_lipids])

    np_scaler = MinMaxScaler()
    X_train_np = np_scaler.fit_transform(X_train[:, num_lipids:])
    X_cand_np = np_scaler.transform(X_candidates[:, num_lipids:])

    X_train_model = np.hstack([X_train_clr, X_train_np])
    X_cand_model = np.hstack([X_cand_clr, X_cand_np])

    if model_type == 'BOOSTER':
        if MultiOutputDeepEnsemble is None:
            raise ImportError("probabilistic_booster_v3.py is required for BOOSTER mode.")

        # Stretch the bounded target onto the real line with a logit link so the
        # ensemble can model it without boundary artefacts.
        y_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        y_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_stretched = logit(y_scaler.fit_transform(y_2d))

        model = MultiOutputDeepEnsemble(
            n_models=20, n_estimators=300, learning_rate=0.01,
            max_depth=8, min_samples_leaf=1, feature_fraction=1.0,
            randomize_depth=True, target_variance=2,
        )
        model.fit(X_train_model, y_stretched, groups=groups)

        pred_stretched, unc_stretched = model.predict(X_cand_model)

        # Invert the logit link and propagate uncertainty through it.
        pred_bounded = expit(pred_stretched)
        pred_real = y_scaler.inverse_transform(pred_bounded)
        derivative = pred_bounded * (1 - pred_bounded)
        scale_factor = (y_scaler.data_max_ - y_scaler.data_min_)
        unc_real = unc_stretched * derivative * scale_factor

        return pred_real.ravel(), unc_real.ravel()

    elif model_type == 'GPR':
        kernel = (C(1.0, (1e-3, 1e3))
                  * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10)))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                         normalize_y=True, random_state=42)
        model.fit(X_train_model, y_train)
        pred, std = model.predict(X_cand_model, return_std=True)
        return pred, std.ravel()

    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------
def compute_nll(y_true, y_pred, y_std):
    """Gaussian negative log-likelihood (lower is better)."""
    y_std = np.maximum(y_std, 1e-6)
    nll = 0.5 * np.log(2 * np.pi) + np.log(y_std) + ((y_true - y_pred) ** 2) / (2 * y_std ** 2)
    return np.mean(nll)


def evaluate_single_fold(train_idx, test_idx, X, y_size, y_pdi, y_ee, groups, model_type):
    """Train on the training groups and predict the single held-out group."""
    X_train, X_test = X[train_idx], X[test_idx]
    groups_train = groups[train_idx]
    targets_map = {'Size': y_size, 'PDI': y_pdi, 'EE': y_ee}

    fold_results = {}
    for name, y_full in targets_map.items():
        pred, std = train_and_predict(
            model_type, X_train, y_full[train_idx], X_test, groups=groups_train
        )
        fold_results[name] = {'true': y_full[test_idx], 'pred': pred, 'std': std}
    return fold_results


def run_loocv(filepath, model_type='BOOSTER', n_jobs=4):
    """Run Leave-One-Group-Out CV and return consolidated predictions per target."""
    X, y_size, y_pdi, y_ee, groups = load_data(filepath)
    print(f"Loaded {len(X)} rows across {len(np.unique(groups))} unique formulations.")

    logo = LeaveOneGroupOut()
    print(f"Running LOO-CV ({model_type}) over {logo.get_n_splits(groups=groups)} folds...")

    results_list = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(evaluate_single_fold)(
            train_idx, test_idx, X, y_size, y_pdi, y_ee, groups, model_type
        )
        for train_idx, test_idx in logo.split(X, y_size, groups)
    )

    consolidated = {k: {'true': [], 'pred': [], 'std': []} for k in ['Size', 'PDI', 'EE']}
    for fold_res in results_list:
        for name in consolidated:
            consolidated[name]['true'].extend(fold_res[name]['true'])
            consolidated[name]['pred'].extend(fold_res[name]['pred'])
            consolidated[name]['std'].extend(fold_res[name]['std'])
    return consolidated


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def report(consolidated, model_type, make_plot=True):
    """Print R2/NLL/MAE per target and optionally show diagnostic plots."""
    metrics = {}
    for name, data in consolidated.items():
        y_t = np.array(data['true'])
        y_p = np.array(data['pred'])
        y_s = np.array(data['std'])
        metrics[name] = {
            'R2': r2_score(y_t, y_p),
            'NLL': compute_nll(y_t, y_p, y_s),
            'MAE': mean_absolute_error(y_t, y_p),
        }
        m = metrics[name]
        print(f"  {name:5s} | R2: {m['R2']:.3f} | NLL: {m['NLL']:.2f} | MAE: {m['MAE']:.3f}")

    if not make_plot:
        return metrics

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, consolidated):
        data = consolidated[name]
        y_t, y_p, y_s = map(np.array, (data['true'], data['pred'], data['std']))
        order = np.argsort(y_t)
        x = np.arange(len(y_t))

        ax.plot(x, y_t[order], 'k-', label='True')
        ax.plot(x, y_p[order], 'r--', label='Pred')
        ax.fill_between(x, y_p[order] - 1.96 * y_s[order], y_p[order] + 1.96 * y_s[order],
                        color='red', alpha=0.2, label='95% CI')
        m = metrics[name]
        ax.set_title(f"{name}\nR2: {m['R2']:.3f} | NLL: {m['NLL']:.2f} | MAE: {m['MAE']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"LOO-CV Diagnostics: {model_type}", fontsize=16)
    fig.tight_layout()
    plt.show()
    return metrics


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ExoAI LOO-CV diagnostics (no GUI).")
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--model", default="BOOSTER", choices=["BOOSTER", "GPR"],
                        help="Surrogate model type.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel CV workers.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the diagnostic plots.")
    args = parser.parse_args()

    consolidated = run_loocv(args.data, model_type=args.model, n_jobs=args.jobs)
    print("\n--- LOO-CV Metrics ---")
    report(consolidated, args.model, make_plot=not args.no_plot)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
