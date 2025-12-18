# tests/test_models.py

import pandas as pd
import numpy as np
from ptrader.models.sk import RFProb
from ptrader.models.lgb import LGBProb
from ptrader.models.ensemble import Ensemble
import pytest
from unittest.mock import MagicMock


@pytest.fixture(autouse=True)
def patch_loggers(monkeypatch):
    # Patch HumanLogger and JSONLLogger globally for all tests
    monkeypatch.setattr("ptrader.cli.main.HumanLogger", lambda path: MagicMock())
    monkeypatch.setattr("ptrader.cli.main.JSONLLogger", lambda path: MagicMock())


def test_rf_and_lgb_predict_proba():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, size=100))
    rf = RFProb().fit(X, y)
    lgb = LGBProb().fit(X, y)
    assert len(rf.predict(X)) == 100
    assert len(lgb.predict(X)) == 100


def test_ensemble_retains_models():
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, size=200))
    ens = Ensemble()
    ens.add(RFProb())
    ens.add(LGBProb())
    ens.fit(X, y)
    preds = ens.predict(X)
    assert len(preds) == 200
