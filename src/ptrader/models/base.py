# ptrader/models/base.py

import numpy as np
import pandas as pd


class Model:
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement predict()")
