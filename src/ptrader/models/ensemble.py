# ptrader/models/ensemble.py

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from .base import Model


class Ensemble:
    def __init__(self):
        self.models = []
        self.cal = IsotonicRegression(out_of_bounds="clip")
        self.meta = LogisticRegression(C=20, max_iter=1000)
        self.retained = []

    def add(self, m: Model) -> None:
        self.models.append(m)

    def fit(self, X, y, val_split=0.2, min_auc=0.55):
        # split train/validation (no shuffle for time series)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, shuffle=False)

        # simple rolling CV on training set
        fold = max(3, min(5, len(X_train) // 500))
        idxs = np.array_split(np.arange(len(X_train)), fold)

        oof = np.zeros((len(X_train), len(self.models)))

        for j, m in enumerate(self.models):
            preds = np.zeros(len(X_train))
            for k in range(1, len(idxs)):
                tr = np.concatenate(idxs[:k])
                va = idxs[k]
                if len(tr) == 0 or len(va) == 0 or y_train.iloc[tr].nunique() < 2:
                    continue
                temp = type(m)()
                try:
                    temp.fit(
                        X_train.iloc[tr],
                        y_train.iloc[tr],
                        eval_set=[(X_train.iloc[va], y_train.iloc[va])],
                        eval_metric="logloss",
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                except TypeError:
                    temp.fit(X_train.iloc[tr], y_train.iloc[tr])
                preds[va] = temp.predict(X_train.iloc[va])
            oof[:, j] = preds

            # compute CV metric for this model
            if np.nanstd(oof[:, j]) > 1e-6 and y_train.nunique() > 1:
                try:
                    auc = roc_auc_score(y_train, oof[:, j])
                except Exception:
                    auc = float("nan")
            else:
                auc = 0.5

            # attach stats to the model instance
            m.cv_stats = {
                "auc": auc,
                "mean_pred": float(np.nanmean(oof[:, j])),
                "std_pred": float(np.nanstd(oof[:, j])),
                "n_samples": int(len(y_train)),
            }

        # prune by AUC
        self.retained = [m for m in self.models if m.cv_stats.get("auc", 0.5) >= min_auc]
        if not self.retained:
            self.retained = self.models[:]
        # Fit retained models on the full X_train/y_train
        for m in self.retained:
            m.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.retained:
            return np.zeros(len(X))

        # collect predictions from retained models
        preds = np.column_stack([m.predict(X) for m in self.retained])
        avg = preds.mean(axis=1)

        if self.cal is not None:
            try:
                cald = np.column_stack(
                    [self.cal.transform(preds[:, j]) for j in range(preds.shape[1])]
                )
                return self.meta.predict_proba(cald)[:, 1] if self.meta else cald.mean(axis=1)
            except Exception:
                return avg

        return avg
