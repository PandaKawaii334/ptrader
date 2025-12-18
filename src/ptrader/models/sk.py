# ptrader/models/sk.py

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore
from .base import Model


class RFProb(Model):
    def __init__(self, name="rf", seed=42):
        self.name = name
        self.seed = seed
        self.scaler = RobustScaler()
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )

    def fit(self, X, y, **kwargs):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
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
