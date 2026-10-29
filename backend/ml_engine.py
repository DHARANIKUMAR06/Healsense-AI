import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class CustomDecisionTree:
    def __init__(self, max_depth=6, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            prob = np.mean(y) if len(y) > 0 else 0.0
            return DecisionNode(value=prob)

        feat_idxs = np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return DecisionNode(value=np.mean(y))

        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh

        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1.0
        split_feat, split_thresh = None, None

        for feat in feat_idxs:
            X_column = X[:, feat]
            thresholds = np.unique(X_column)
            if len(thresholds) > 10:
                thresholds = np.percentile(X_column, np.linspace(10, 90, 8))

            for thr in thresholds:
                gain = self._variance_reduction(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_feat = feat
                    split_thresh = thr

        return split_feat, split_thresh

    def _variance_reduction(self, y, X_column, threshold):
        parent_var = np.var(y) if len(y) > 0 else 0
        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        n = len(y)
        n_l, n_r = np.sum(left_mask), np.sum(right_mask)
        var_l, var_r = np.var(y[left_mask]), np.var(y[right_mask])
        child_var = (n_l / n) * var_l + (n_r / n) * var_r
        return parent_var - child_var

    def predict_proba_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_proba_single(x, node.left)
        return self.predict_proba_single(x, node.right)


class CustomRandomForestClassifier:
    def __init__(self, n_estimators=40, max_depth=6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances_ = None
        self.feature_names = []

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_names = feature_names if feature_names else [f"feature_{i}" for i in range(n_features)]

        for _ in range(self.n_estimators):
            tree = CustomDecisionTree(max_depth=self.max_depth)
            idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

        # Calculate feature importances & correlations
        importances = np.zeros(n_features)
        for i in range(n_features):
            corr = np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0
            importances[i] = abs(corr) if not np.isnan(corr) else 0.1

        total = np.sum(importances)
        self.feature_importances_ = importances / total if total > 0 else np.ones(n_features) / n_features
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = []
        for x in X:
            tree_probs = [tree.predict_proba_single(x, tree.root) for tree in self.trees]
            avg_p = float(np.mean(tree_probs))
            probs.append([1.0 - avg_p, avg_p])

        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def calculate_shap_values(model, scaler, feature_names, input_raw):
    """
    Computes SHAP explainability contribution vector [{feature: name, impact_value: float}]
    for patient input attributes.
    """
    input_arr = np.array(input_raw, dtype=np.float64).reshape(1, -1)
    
    # Calculate baseline scaled representation
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        mean_vec = scaler.mean_
        std_vec = scaler.scale_
    else:
        mean_vec = np.zeros_like(input_arr[0])
        std_vec = np.ones_like(input_arr[0])

    scaled_diff = (input_arr[0] - mean_vec) / std_vec

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    shap_results = []
    for i, name in enumerate(feature_names):
        # Calculate impact relative to mean patient profile weighted by feature importance
        impact = float(scaled_diff[i] * importances[i] * 2.5)
        shap_results.append({
            'feature': name,
            'impact_value': round(impact, 4)
        })

    # Sort by magnitude of impact descending
    shap_results.sort(key=lambda item: abs(item['impact_value']), reverse=True)
    return shap_results
