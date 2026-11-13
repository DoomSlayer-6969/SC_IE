from sc_ie import config


def test_robust_term_condition():
    # Eq. 21: 2 rho^2 >= r
    assert 2 * config.RHO**2 >= config.R


def test_legendre_basis_size():
    # Eq. 33: phi0 = [1, I1, I2, I3, I4]
    assert config.LEGENDRE_ORDER + 1 == 5


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    import importlib

    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    assert list(tmp_path.iterdir()) == []
