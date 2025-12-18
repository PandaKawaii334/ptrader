# ptrader/models/lgb.py

import lightgbm as lgb
from .base import Model


class LGBProb(Model):
    def __init__(self, name="lgb", seed=42):
        self.name = name
        self.seed = seed
        self.model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            min_child_samples=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_estimators=800,
            random_state=seed,
            is_unbalance=True,
            verbose=-1,
        )

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        Xs = X.copy()
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xs)
            # If only one class was seen during training, proba has shape (n,1)
            if proba.shape[1] == 1:
                # Return that single column (all probabilities for the observed class)
                return proba.ravel()
            return proba[:, 1]  # normal case: probability of positive class
        # Fallback to hard predictions
        return self.model.predict(Xs).astype(float)
