import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge

class MultiOutputDeepEnsemble: # Not actually multi-output
    def __init__(self, n_models=5, n_estimators=100, learning_rate=0.1, min_samples_leaf=4,
                 max_depth=5, feature_fraction=1.0, randomize_depth=True, target_variance=None):
        self.params = {
            'n_models': n_models,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'min_samples_leaf': min_samples_leaf,
            'max_depth': max_depth,
            'feature_fraction': feature_fraction,
            'randomize_depth': randomize_depth,
            'target_variance': target_variance
        }
        self.output_ensembles = [] # Stores a list of model-sets for each output column

    def _compute_gradients(self, y, mu, log_var):
        var = np.maximum(np.exp(log_var), 1e-6)
        grad_mu = -(y - mu) / var
        hess_mu = 1.0 / var
        grad_var = 0.5 - 0.5 * ((y - mu)**2) / var
        hess_var = 0.5 * ((y - mu)**2) / var + 1e-6 
        return grad_mu, hess_mu, grad_var, hess_var

    def _train_ensemble_for_target(self, X, y, groups):
        """Trains a full Deep Ensemble for a single scalar target."""
        models = []
        n_models = self.params['n_models']
        
        # Variance bounds
        if self.params['target_variance'] is not None:
            min_log = np.log(max(self.params['target_variance'] / 100, 1e-6))
            max_log = np.log(self.params['target_variance'] * 10)
        else:
            min_log, max_log = -5.0, 5.0

        for k in range(n_models):
            rng = np.random.RandomState(42 + k)
            indices = rng.choice(len(y), len(y), replace=True)
            X_b, y_b = X[indices], y[indices]

            # Kernel RBF Base
            init_model = KernelRidge(kernel='rbf', alpha=1.0)
            init_model.fit(X_b, y_b)
            base_mu = init_model.predict(X_b)
            base_log_var = np.log(max(np.var(y_b - base_mu), 1e-6))

            curr_mu = base_mu.copy()
            curr_log_var = np.full(len(y_b), base_log_var)
            
            trees_mu, trees_var = [], []
            
            for _ in range(self.params['n_estimators']):
                g_mu, h_mu, g_var, h_var = self._compute_gradients(y_b, curr_mu, curr_log_var)
                
                # --- CORRECTION START: Apply Randomize Depth ---
                current_max_depth = self.params['max_depth']
                if self.params['randomize_depth']:
                    # Randomly choose a depth between 2 and max_depth
                    current_max_depth = rng.randint(2, self.params['max_depth'] + 1)
                
                # --- CORRECTION START: Apply Feature Fraction ---
                # Pass feature_fraction to max_features
                
                # Mean Tree
                t_mu = DecisionTreeRegressor(
                    max_depth=current_max_depth, 
                    max_features=self.params['feature_fraction'], # Applied here
                    random_state=rng
                )
                t_mu.fit(X_b, -g_mu / (h_mu + 1.0), sample_weight=(h_mu + 1.0))
                trees_mu.append(t_mu)
                curr_mu += self.params['learning_rate'] * t_mu.predict(X_b)
                
                # Var Tree (Keep slightly simpler than mu tree)
                var_depth = max(1, current_max_depth - 2)
                t_var = DecisionTreeRegressor(
                    max_depth=var_depth, 
                    max_features=self.params['feature_fraction'], # Applied here
                    random_state=rng
                )
                t_var.fit(X_b, -g_var / (h_var + 1.0), sample_weight=(h_var + 1.0))
                trees_var.append(t_var)
                curr_log_var = np.clip(curr_log_var + self.params['learning_rate'] * t_var.predict(X_b), min_log, max_log)

            models.append({
                'init_model': init_model, 'trees_mu': trees_mu, 
                'trees_var': trees_var, 'base_log_var': base_log_var,
                'min_log': min_log, 'max_log': max_log
            })
        return models

    def fit(self, X, Y, groups=None):
        """Y can now be (n_samples, n_outputs)"""
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n_outputs = Y.shape[1]
        self.output_ensembles = []
        
        for i in range(n_outputs):
            #print(f"Training ensemble for Output {i}...")
            ensemble = self._train_ensemble_for_target(X, Y[:, i], groups)
            self.output_ensembles.append(ensemble)

    def predict(self, X):
        """Returns means and sigmas of shape (n_samples, n_outputs)"""
        all_means = []
        all_sigmas = []

        for ensemble in self.output_ensembles:
            mu_preds = np.zeros((self.params['n_models'], len(X)))
            var_preds = np.zeros((self.params['n_models'], len(X)))
            
            for k, model in enumerate(ensemble):
                curr_mu = model['init_model'].predict(X)
                curr_log_var = np.full(len(X), model['base_log_var'])
                
                for t in model['trees_mu']: curr_mu += self.params['learning_rate'] * t.predict(X)
                for t in model['trees_var']: curr_log_var += self.params['learning_rate'] * t.predict(X)
                
                mu_preds[k, :] = curr_mu
                var_preds[k, :] = np.exp(np.clip(curr_log_var, model['min_log'], model['max_log']))

            final_mu = np.mean(mu_preds, axis=0)
            # Total Variance = Epistemic (variance of means) + Aleatoric (mean of variances)
            final_sigma = np.sqrt(np.var(mu_preds, axis=0) + np.mean(var_preds, axis=0))
            
            all_means.append(final_mu)
            all_sigmas.append(final_sigma)

        return np.column_stack(all_means), np.column_stack(all_sigmas)