import numpy as np
import pytest

from sc_ie import config as cfg
from sc_ie import systems as S
from sc_ie.integrate import largest_lyapunov

rng = np.random.default_rng(0)
a1, b1, c1 = cfg.LU_A1, cfg.LU_B1, cfg.LU_C1
a2, b2, c2 = cfg.GT_A2, cfg.GT_B2, cfg.GT_C2


def test_lu_matches_eq37():
    x = rng.normal(size=3)
    x1, x2, x3 = x
    expected = [a1 * (x2 - x1), -x1 * x3 + b1 * x2, x1 * x2 - c1 * x3]
    assert np.allclose(S.lu_rhs(0.0, x), expected)


def test_genesio_tesi_matches_eq38():
    y = rng.normal(size=3)
    y1, y2, y3 = y
    expected = [y2, y3, -a2 * y1 - b2 * y2 - c2 * y3 + y1**2]
    assert np.allclose(S.genesio_tesi_rhs(0.0, y), expected)


def test_delta_A_is_eq40_as_printed():
    expected = [[-a1, 1 - a1, 0], [0, -b1, 1], [-a2, -b2, -(c2 - c1)]]
    assert np.allclose(S.delta_A(), expected)


def test_eq40_differs_from_eq38_only_at_first_entry():
    # Documents the paper inconsistency: with dA[0,0] = -a1 as printed,
    # (A + dA)[0,0] = -2 a1 instead of 0 (Eq. 38). Everything else matches.
    A_resp = S.lu_A() + S.delta_A()
    A_eq38 = np.array([[0, 1, 0], [0, 0, 1], [-a2, -b2, -c2]])
    diff = A_resp - A_eq38
    assert diff[0, 0] == pytest.approx(-2 * a1)
    diff[0, 0] = 0.0
    assert np.allclose(diff, 0.0)


def test_response_rhs_is_eq40():
    y, u = rng.normal(size=3), rng.normal(size=3)
    expected = (S.lu_A() + S.delta_A()) @ y + np.array([0, 0, y[0] ** 2]) + u
    assert np.allclose(S.response_rhs(0.0, y, u), expected)


def test_error_dynamics_eq42():
    # E' = A E + f(t) + B u  with  f = dA Y + g(Y) - g(X)
    x, y, u = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    e_dot = S.response_rhs(0.0, y, u) - S.lu_rhs(0.0, x)
    assert np.allclose(e_dot, S.lu_A() @ (y - x) + S.sync_uncertainty(x, y) + u)


def test_chua_diode_piecewise_linear():
    a, b = cfg.CHUA_A, cfg.CHUA_B
    inner = np.linspace(-0.9, 0.9, 7)
    assert np.allclose(S.chua_h(inner), a * inner)
    outer = np.array([1.5, 3.0])
    assert np.allclose(S.chua_h(outer), b * outer + (a - b))
    assert np.allclose(S.chua_h(-outer), -(b * outer + (a - b)))


def test_chua_matches_eq59():
    t, x = 0.7, rng.normal(size=3)
    x1, x2, x3 = x
    p, q, g = cfg.CHUA_P, cfg.CHUA_Q, cfg.CHUA_GAMMA
    expected = [p * (x2 - x1 - S.chua_h(x1)) + np.cos(t), x1 - x2 + x3, -q * x2 - g * x3]
    assert np.allclose(S.chua_rhs(t, x), expected)


def test_lu_is_chaotic():
    assert largest_lyapunov(S.lu_rhs, cfg.LU_X0, 1e-3, 50.0, t_transient=10.0) > 1.0


@pytest.mark.parametrize("gamma, chaotic", [(0.0385, True), (0.385, False)])
def test_chua_chaos_depends_on_gamma(gamma, chaotic):
    A = S.chua_A(gamma=gamma)
    lle = largest_lyapunov(lambda t, x: S.chua_rhs(t, x, A), cfg.CHUA_X0, 1e-3, 100.0)
    assert (lle > 0.2) if chaotic else (lle < 0.05)
