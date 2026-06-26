import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import joblib
from joblib import Parallel, delayed

# Import your custom surrogate model
# Use the bundled surrogate model
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probabilistic_booster import MultiOutputDeepEnsemble

class PairwiseProbabilisticRanker:
    def __init__(self, confidence_threshold=0.50):
        self.model = None
        self.confidence_threshold = confidence_threshold
        
    def _clr_transform(self, X):
        X_mat = np.array(X, dtype=float)
        log_x = np.log(X_mat + 1e-8)
        gm = np.mean(log_x, axis=1, keepdims=True)
        return log_x - gm

    def _calculate_desirability(self, size, pdi, ee, bin_size=10.0):
        size_score = np.clip((180 - size) / (180 - 80), 0, 1)
        pdi_score = np.clip((0.4 - pdi) / (0.4 - 0.05), 0, 1)
        ee_score = np.clip((ee - 80) / (100 - 80), 0, 1)
        
        raw_score = ((size_score * pdi_score * ee_score) ** (1/3)) * 100.0
        binned_score = np.round(raw_score / bin_size) * bin_size
        return binned_score

    def generate_pairs(self, X, y_raw):
        scores = self._calculate_desirability(y_raw[:, 0], y_raw[:, 1], y_raw[:, 2])
        X_clr = self._clr_transform(X)
        
        n_samples = len(X)
        X_pairs = []
        Y_diffs = []
        
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    X_pairs.append(X_clr[i] - X_clr[j])
                    Y_diffs.append(scores[i] - scores[j])
                    
        return np.array(X_pairs), np.array(Y_diffs).reshape(-1, 1)

    def fit(self, X_pairs, Y_diffs):
        self.model = MultiOutputDeepEnsemble(
            n_models=10, n_estimators=500, learning_rate=0.01,     
            max_depth=6, min_samples_leaf=2, feature_fraction=1.0,   
            randomize_depth=True, target_variance=1
        )
        self.model.fit(X_pairs, Y_diffs)

    def predict_pair(self, X_A, X_B):
        X_A_clr = self._clr_transform(X_A)
        X_B_clr = self._clr_transform(X_B)
        
        X_pair = X_A_clr - X_B_clr
        mu_diff, sigma_diff = self.model.predict(X_pair)
        
        classifications = []
        probabilities = []
        
        for i in range(len(mu_diff)):
            m = mu_diff[i, 0]
            s = np.maximum(sigma_diff[i, 0], 1e-6)
            
            # Probability that Score A > Score B
            prob_A_better = norm.cdf(m / s) 
            probabilities.append(prob_A_better)
            
            if prob_A_better > self.confidence_threshold:
                classifications.append(1)  # A Wins
            elif prob_A_better < (1.0 - self.confidence_threshold):
                classifications.append(-1) # B Wins
            else:
                classifications.append(0)  # Tie (Uncertain)
                
        return np.array(classifications), np.array(probabilities)

# --- NEW PARALLEL WORKER FUNCTION ---
def _process_fold(train_idx, test_idx, X, y, threshold):
    """Processes a single fold entirely independently to avoid thread collisions."""
    ranker = PairwiseProbabilisticRanker(confidence_threshold=threshold)
    
    X_train_raw, y_train_raw = X[train_idx], y[train_idx]
    X_test_raw, y_test_raw = X[test_idx], y[test_idx]
    
    # Train
    X_train_pairs, Y_train_diffs = ranker.generate_pairs(X_train_raw, y_train_raw)
    ranker.fit(X_train_pairs, Y_train_diffs)
    
    # Test
    X_test_pairs, Y_test_diffs = ranker.generate_pairs(X_test_raw, y_test_raw)
    
    if len(X_test_pairs) > 0:
        X_test_A = X_test_raw.repeat(len(X_test_raw), axis=0)
        X_test_B = np.tile(X_test_raw, (len(X_test_raw), 1))
        
        valid_idx = ~np.all(X_test_A == X_test_B, axis=1)
        X_test_A = X_test_A[valid_idx]
        X_test_B = X_test_B[valid_idx]
        
        preds, probs = ranker.predict_pair(X_test_A, X_test_B)
        return Y_test_diffs.flatten(), preds.flatten(), probs.flatten()
    return np.array([]), np.array([]), np.array([])

def visualize_results(y_true, y_pred, y_prob):
    print("\nGenerating Validation Visualization...")
    
    y_true_class = np.where(y_true > 0.005, 1, np.where(y_true < -0.005, -1, 0))
    
    # Calculate Accuracy (excluding True Ties)
    mask_acc = y_true_class != 0
    acc = np.mean(y_true_class[mask_acc] == y_pred[mask_acc]) if len(y_true_class[mask_acc]) > 0 else 0.0
    
    # Calculate ROC-AUC (Isolating clear Win/Loss scenarios)
    y_true_binary = np.where(y_true_class[mask_acc] == 1, 1, 0)
    y_prob_binary = y_prob[mask_acc]
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
    roc_auc = auc(fpr, tpr)
    
    # Setup Side-by-Side Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confusion Matrix
    labels = [-1, 0, 1]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true_class, y_pred):
        try:
            cm[labels.index(t), labels.index(p)] += 1
        except ValueError:
            pass

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Pred B Wins', 'Pred Tie', 'Pred A Wins'],
                yticklabels=['True B Wins', 'True Tie', 'True A Wins'])
    ax1.set_title(f'Confusion Matrix\nAccuracy (Strict Win/Loss): {acc:.1%}')
    ax1.set_ylabel('Actual Relationship')
    ax1.set_xlabel('Predicted Relationship')
    
    # 2. ROC Curve
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (A Wins vs B Wins)')
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('pairwise_cv_results_with_auc.png', dpi=300)
    plt.show()

def main():
    print("1. Loading ExoAI_AL.csv...")
    df = pd.read_csv('ExoAI_AL.csv')  # or 'sample_data/seed_24.csv' for demo
    LIPID_KEYS = ['dc_chol', 'chol', 'dssm', 'dspc', 'dsps', 'dope']
    X = df[LIPID_KEYS].values
    y = df[['particle_size', 'pdi', 'EE']].values
    
    # Keeping threshold strictly at 0.50 for decisiveness 
    threshold = 0.50
    
    print("\n2. Running 12-Fold Cross Validation in PARALLEL...")
    kf = KFold(n_splits=12, shuffle=True, random_state=42)
    
    # Parallel Execution (-1 uses all available CPU cores)
    results = Parallel(n_jobs=18, verbose=10)(
        delayed(_process_fold)(train_idx, test_idx, X, y, threshold)
        for train_idx, test_idx in kf.split(X)
    )
    
    # Unpack parallel results
    all_y_true, all_y_pred, all_y_prob = [], [], []
    for y_t, y_p, y_pr in results:
        all_y_true.extend(y_t)
        all_y_pred.extend(y_p)
        all_y_prob.extend(y_pr)
        
    print("\n   ✓ CV Completed.")
    
    # Show Visualizations
    visualize_results(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob))

    print("\n3. Training Final Classifier on ALL data...")
    final_ranker = PairwiseProbabilisticRanker(confidence_threshold=threshold)
    X_all_pairs, Y_all_diffs = final_ranker.generate_pairs(X, y)
    final_ranker.fit(X_all_pairs, Y_all_diffs)
    
    print("\n4. Saving model weights...")
    joblib.dump(final_ranker, 'ExoAI_PairwiseClassifier.joblib')
    print("   ✓ Saved as 'ExoAI_PairwiseClassifier.joblib'")

if __name__ == "__main__":
    main()