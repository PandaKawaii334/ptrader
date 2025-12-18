# ptrader/labels/adaptive.py

import numpy as np
import pandas as pd


def adaptive_labels(
    df: pd.DataFrame,
    base_horizons: tuple[int, ...] = (2, 4, 8),
    fee: float = 0.001,
    slip: float = 0.0005,
) -> pd.Series:
    vol = df["close"].pct_change().rolling(16).std().fillna(0)
    idx = np.clip((vol / (vol.rolling(128).median() + 1e-10)).fillna(1.0), 0.5, 2.0)
    chosen_h = np.array(base_horizons)[
        np.floor((idx - 0.5) / (1.5 / len(base_horizons)))
        .astype(int)
        .clip(0, len(base_horizons) - 1)
    ]
    net_cost = 2 * (fee + slip)

    rets = {h: df["close"].shift(-h) / df["close"] - 1 for h in sorted(set(base_horizons))}
    labels: list[int] = []
    for i in range(len(df)):
        h = int(chosen_h[i])
        r = rets[h].iloc[i] if i + h < len(df) else np.nan
        labels.append(int((r - net_cost) >= (0.75 * net_cost)))

    max_h = max(base_horizons)
    return pd.Series(labels[:-max_h], dtype="int64").reset_index(drop=True)
