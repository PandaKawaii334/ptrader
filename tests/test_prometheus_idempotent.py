from ptrader.metrics.prometheus import make_prom


def test_make_prom_idempotent():
    g1, c1 = make_prom()
    g2, c2 = make_prom()
    # Should return the same dict objects (cached) and not raise
    assert g1 is g2
    assert c1 is c2
    # Expected gauge keys exist
    expected = {"equity", "position", "price", "signal"}
    assert expected.issubset(set(g1.keys()))
    assert "kills" in c1
