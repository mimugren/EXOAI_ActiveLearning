"""
ExoAI — Active-Learning Acquisition
===================================

Standalone backbone (no GUI) that proposes the next batch of experiments for the
ExoAI lipid-nanoparticle optimization pipeline.

Pipeline
--------
1. Train a probabilistic surrogate (BOOSTER or GPR) for each target
   (particle Size, PDI, EE) on all available data.
2. Generate a large, spatially diverse pool of candidate formulations that
   respect user-defined per-lipid composition bounds.
3. Score every candidate with a desirability objective and an uncertainty
   estimate, then combine them into an Upper-Confidence-Bound style
   "Optimistic Potential" (score + kappa * uncertainty).
4. Select a diverse batch:
      * EXPLOIT — highest Optimistic Potential
      * EXPLORE — highest model uncertainty
   Kappa decays automatically as more data accumulates (explore -> exploit).

Usage
-----
    python acquisition.py --data my_data.csv --model BOOSTER --out next_batch.csv

CSV requirements
----------------
Columns: dc_chol, chol, dssm, dspc, dsps, dope, NP, particle_size, pdi, EE
"""

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler

try:
    from probabilistic_booster_v3 import MultiOutputDeepEnsemble
except ImportError:
    MultiOutputDeepEnsemble = None

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
LIPID_KEYS = ['dc_chol', 'chol', 'dssm', 'dspc', 'dsps', 'dope']
FEATURES = LIPID_KEYS + ['NP']

# Desirability targets used to score each candidate formulation.
SCORING_PARAMS = {
    'size_min': 100.0, 'size_target': 130.0, 'size_max': 180.0,
    'pdi_accept': 0.15, 'pdi_fail': 0.3,
    'ee_fail': 95.0, 'ee_accept': 99.0,
}

# Per-lipid composition bounds (min %, max %). Candidates must respect these.
DEFAULT_BOUNDS = {
    'dc_chol': (20, 55), 'chol': (5, 30), 'dssm': (5, 25),
    'dspc': (5, 25), 'dsps': (5, 15), 'dope': (5, 25),
}

NP_FIXED = 6.0                 # N/P ratio held constant for generated candidates
TARGET_DECAY_GENTLE = 0.7      # desirability decay shape inside the size window


# ----------------------------------------------------------------------
# Pre-processing + model (shared with the LOO-CV script)
# ----------------------------------------------------------------------
def clr_transform(X):
    """Centered Log-Ratio transform for compositional (lipid %) data."""
    X_mat = np.array(X, dtype=float)
    log_x = np.log(X_mat + 1e-8)
    gm = np.mean(log_x, axis=1, keepdims=True)
    return log_x - gm


def load_data(filepath):
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
            df['Group_ID'].values,
            len(df))


def train_and_predict(model_type, X_train, y_train, X_candidates, groups=None):
    """Train a surrogate and return (mean, std) predictions on candidates."""
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
# Desirability scoring
# ----------------------------------------------------------------------
def get_desirability_score(value, target_type, limits, curvature=1.0):
    """Map a measured value to a [0, 1] desirability."""
    if target_type == 'min':                       # smaller is better
        acc, fail = limits
        if value < acc:
            return 1.0
        if value > fail:
            return 0.0
        return ((fail - value) / (fail - acc)) ** curvature

    if target_type == 'max':                        # larger is better
        fail, acc = limits
        if value > acc:
            return 1.0
        if value < fail:
            return 0.0
        return ((value - fail) / (acc - fail)) ** curvature

    if target_type == 'target':                     # a window is best
        low, target, high = limits
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        if value <= target:
            d = 1.0 - TARGET_DECAY_GENTLE * ((value - low) / (target - low))
        else:
            d = (1.0 - TARGET_DECAY_GENTLE) * ((high - value) / (high - target))
        return d ** curvature
    return 0.0


def calculate_objective(row):
    """Geometric mean of the Size / PDI / EE desirabilities for one candidate."""
    if row.isnull().any():
        return 0.0
    p = SCORING_PARAMS
    d_size = get_desirability_score(row['particle_size'], 'target',
                                    (p['size_min'], p['size_target'], p['size_max']), curvature=2.0)
    d_pdi = get_desirability_score(row['pdi'], 'min',
                                   (p['pdi_accept'], p['pdi_fail']), curvature=1.0)
    d_ee = get_desirability_score(row['EE'], 'max',
                                  (p['ee_fail'], p['ee_accept']), curvature=2.0)
    return (d_size * d_pdi * d_ee) ** (1.0 / 3.0)


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------
def generate_candidates(bounds, n_samples=50000, grid_spacing=1.5, seed=0):
    """
    Sample diverse lipid compositions (summing to 100%) within `bounds`.

    A Dirichlet mix gives both balanced and "spiky" recipes; a hash-voxel grid
    enforces a minimum spacing so the pool explores the design space evenly.
    """
    rng = np.random.default_rng(seed)
    bounds_array = np.array([bounds[k] for k in LIPID_KEYS])
    min_b, max_b = bounds_array[:, 0], bounds_array[:, 1]
    n_dims = len(LIPID_KEYS)

    samples, seen_voxels = [], set()
    for _ in range(500):                            # iteration cap (safety)
        if len(samples) >= n_samples:
            break
        balanced = rng.dirichlet(np.ones(n_dims), size=10000)
        spiky = rng.dirichlet(np.ones(n_dims) * 0.5, size=5000)
        raw = np.vstack([balanced, spiky]) * 100

        in_bounds = np.all((raw >= min_b) & (raw <= max_b), axis=1)
        valid = raw[in_bounds]
        if len(valid) == 0:
            continue

        voxels = np.round(valid / grid_spacing).astype(int)
        for i in range(len(voxels)):
            v = tuple(voxels[i])
            if v not in seen_voxels:
                seen_voxels.add(v)
                samples.append(valid[i])
                if len(samples) >= n_samples:
                    break

    final = np.array(samples)
    rng.shuffle(final)
    np_col = np.full((len(final), 1), NP_FIXED)
    print(f"Generated {len(final)} spatially diverse candidates.")
    return np.hstack([final, np_col])


def distinct_round_to_sum(row, bounds, target_sum=100):
    """Round lipid fractions to integers while keeping the sum at 100 and respecting bounds."""
    floats = row.values.astype(float)
    floored = np.floor(floats).astype(int)
    remainder = int(target_sum - floored.sum())
    order = np.argsort(floats - floored)[::-1]
    for i in range(remainder):
        floored[order[i]] += 1

    bounds_array = np.array([bounds[k] for k in row.index])
    min_b, max_b = bounds_array[:, 0], bounds_array[:, 1]

    if not np.all((floored >= min_b) & (floored <= max_b)):
        floored = np.floor(floats).astype(int)
        under = floored < min_b
        floored[under] = min_b[under]
        remainder = int(target_sum - floored.sum())
        if remainder > 0:
            order = np.argsort(max_b - floored)[::-1]
            for i in range(remainder):
                idx = order[i % len(floored)]
                if floored[idx] < max_b[idx]:
                    floored[idx] += 1
        elif remainder < 0:
            order = np.argsort(floored - min_b)[::-1]
            for i in range(abs(remainder)):
                idx = order[i % len(floored)]
                if floored[idx] > min_b[idx]:
                    floored[idx] -= 1
    return pd.Series(floored, index=row.index)


# ----------------------------------------------------------------------
# Diverse batch selection
# ----------------------------------------------------------------------
def get_diverse_top_k(df, features, sort_col, ascending, k=3, min_dist=3.0):
    """Greedily pick the top-k rows by `sort_col` that are >= min_dist apart."""
    df_sorted = df.sort_values(by=sort_col, ascending=ascending)
    selected_idx, selected_vecs = [], []
    for idx, row in df_sorted.iterrows():
        vec = row[features].values.astype(float)
        if not selected_vecs:
            selected_idx.append(idx)
            selected_vecs.append(vec)
        else:
            dists = np.linalg.norm(np.array(selected_vecs) - vec, axis=1)
            if np.min(dists) >= min_dist:
                selected_idx.append(idx)
                selected_vecs.append(vec)
        if len(selected_idx) == k:
            break
    if len(selected_idx) < k:                       # relax the distance rule if needed
        remaining = df_sorted.drop(selected_idx).head(k - len(selected_idx))
        selected_idx.extend(remaining.index)
    return df_sorted.loc[selected_idx].copy()


# ----------------------------------------------------------------------
# Main acquisition routine
# ----------------------------------------------------------------------
def run_acquisition(filepath, model_type='BOOSTER', bounds=None,
                    n_candidates=50000, k_each=3, out_csv="next_batch.csv"):
    bounds = bounds or DEFAULT_BOUNDS
    X_train, y_size, y_pdi, y_ee, groups, len_df = load_data(filepath)

    # Auto-kappa: high exploration early, decaying toward exploitation as data grows.
    # (Seed of 72 reflects the initial design; tune to your own campaign.)
    al_round = max(0, (len_df - 72) / 6.0)
    kappa = 2.0 * (0.95 ** al_round)
    print(f"Data points: {len_df} | round ~{al_round:.1f} | kappa: {kappa:.4f}")

    X_candidates = generate_candidates(bounds, n_samples=n_candidates)

    # Train one surrogate per target in parallel.
    tasks = [(y_size, 'Size'), (y_pdi, 'PDI'), (y_ee, 'EE')]
    results = Parallel(n_jobs=3, backend="multiprocessing")(
        delayed(train_and_predict)(model_type, X_train, y, X_candidates, groups=groups)
        for y, _ in tasks
    )
    (p_size, u_size), (p_pdi, u_pdi), (p_ee, u_ee) = results
    print("Surrogates trained.")

    df = pd.DataFrame(X_candidates, columns=FEATURES)
    df['particle_size'], df['pdi'], df['EE'] = p_size, p_pdi, p_ee
    df['Predicted_Score'] = df.apply(calculate_objective, axis=1)

    # Combine the three per-target uncertainties (normalized by data spread).
    df['Uncertainty'] = np.sqrt((u_size / y_size.std()) ** 2 +
                                (u_pdi / y_pdi.std()) ** 2 +
                                (u_ee / y_ee.std()) ** 2)
    span = df['Uncertainty'].max() - df['Uncertainty'].min()
    df['Uncertainty_Norm'] = ((df['Uncertainty'] - df['Uncertainty'].min()) / span
                              if span > 1e-6 else 0.0)

    # UCB-style acquisition: exploit good scores, with a bonus for uncertainty.
    df['Optimistic_Potential'] = df['Predicted_Score'] + kappa * df['Uncertainty_Norm']

    # Snap compositions to integer percentages that still sum to 100.
    df[LIPID_KEYS] = df[LIPID_KEYS].apply(lambda r: distinct_round_to_sum(r, bounds), axis=1)

    # EXPLOIT: best optimistic potential, kept diverse.
    df_exploit = get_diverse_top_k(df, LIPID_KEYS, 'Optimistic_Potential',
                                   ascending=False, k=k_each)
    df_exploit['Strategy'] = f'Exploit ({model_type})'

    # EXPLORE: highest uncertainty, excluding anything already exploited.
    chosen = set(tuple(r) for r in df_exploit[LIPID_KEYS].values)
    mask = df[LIPID_KEYS].apply(lambda x: tuple(x) not in chosen, axis=1)
    df_explore = get_diverse_top_k(df[mask], LIPID_KEYS, 'Uncertainty',
                                   ascending=False, k=k_each)
    df_explore['Strategy'] = f'Explore ({model_type})'

    next_batch = pd.concat([df_exploit, df_explore])
    cols = ['Strategy'] + FEATURES + ['particle_size', 'pdi', 'EE',
                                      'Predicted_Score', 'Uncertainty_Norm', 'Optimistic_Potential']
    next_batch[cols].to_csv(out_csv, index=False)
    print(f"\nSaved {len(next_batch)} proposed experiments to {out_csv}\n")
    print(next_batch[['Strategy'] + FEATURES +
                     ['Predicted_Score', 'Uncertainty_Norm']].to_string(index=False))
    return next_batch


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ExoAI active-learning acquisition (no GUI).")
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--model", default="BOOSTER", choices=["BOOSTER", "GPR"],
                        help="Surrogate model type.")
    parser.add_argument("--candidates", type=int, default=50000,
                        help="Number of candidate formulations to generate.")
    parser.add_argument("--k", type=int, default=3,
                        help="How many exploit + explore picks to return (each).")
    parser.add_argument("--out", default="next_batch.csv", help="Output CSV path.")
    args = parser.parse_args()

    run_acquisition(args.data, model_type=args.model,
                    n_candidates=args.candidates, k_each=args.k, out_csv=args.out)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
