import numpy as np
import pytest

from sc_ie import basis
from sc_ie import config as cfg
from sc_ie import controller as C
from sc_ie import systems as S
from sc_ie.integrate import rk4


def test_pole_placement_sync():
    A, B = S.lu_A(), np.eye(3)
    k = C.place(A, B, cfg.SYNC_POLES)
    assert np.allclose(np.sort(np.linalg.eigvals(A - B @ k).real), np.sort(cfg.SYNC_POLES))


def test_pole_placement_observer():
    A, Cm = S.chua_A(), S.CHUA_C.reshape(1, 3)
    L = C.place(A.T, Cm.T, cfg.OBSERVER_POLES).T
    assert np.allclose(np.sort(np.linalg.eigvals(A - L @ Cm).real), np.sort(cfg.OBSERVER_POLES))


def test_riccati_scalar_matches_eq20():
    K, Q, rho, r = 3.0, 1.0, 0.5, 0.08
    P = C.riccati_P(np.array([[-K]]), np.eye(1), np.array([[Q]]), rho, r)[0, 0]
    assert -2 * K * P + Q + P**2 * (1 / rho**2 - 2 / r) == pytest.approx(0.0, abs=1e-12)
    assert P > 0


def test_riccati_matrix_residual_and_definiteness():
    A, B = S.lu_A(), np.eye(3)
    A_c = A - B @ C.place(A, B, cfg.SYNC_POLES)
    Q = cfg.Q_SCALE * np.eye(3)
    P = C.riccati_P(A_c, B, Q)
    c = 1 / cfg.RHO**2 - 2 / cfg.R
    residual = A_c.T @ P + P @ A_c + Q + c * P @ B @ B.T @ P
    assert np.allclose(residual, 0.0, atol=1e-10)
    assert np.allclose(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)


def test_riccati_rejects_eq21_violation():
    with pytest.raises(ValueError):
        C.riccati_P(-np.eye(1), np.eye(1), np.eye(1), rho=0.1, r=0.08)


def test_amygdala_weights_never_decrease():
    phi = basis.legendre_regressor(0.3)
    for s in ([1.0, -2.0, 0.5], [-1.0, -1.0, -1.0]):
        dV, dW = C.adaptation(phi, np.array(s), alpha=1.0, beta=1.0)
        # dV = phi^T max(s, 0): sign follows phi, zero where s <= 0
        assert np.allclose(dV, phi.T @ np.maximum(s, 0))
        assert np.allclose(dW, -phi.T @ np.array(s))


def _toy_rms(alpha, beta, use_estimator, T=40.0, dt=2e-3):
    """E' = A E + f(t) + u with f in span of P0..P2(sin t); returns late RMS of E."""
    A, B = S.lu_A(), np.eye(3)
    k = C.place(A, B, [-5.0] * 3)
    P = C.riccati_P(A - B @ k, B, np.eye(3))

    def f_true(t):
        x = np.sin(t)
        return np.array([20 + 10 * x, -15 + 5 * (3 * x * x - 1) / 2, 8 * x])

    def rhs(t, z):
        E, V, W = z[:3], z[3:18], z[18:]
        phi = basis.legendre_regressor(t, omega=1.0)
        s = C.s_signal(E, P, B)
        f_hat = C.estimate(phi, V, W) if use_estimator else np.zeros(3)
        u = C.sync_control(E, k, f_hat, C.robust_term(s))
        dV, dW = C.adaptation(phi, s, alpha, beta) if use_estimator else (0 * V, 0 * W)
        return np.concatenate([A @ E + f_true(t) + u, dV, dW])

    t, Z = rk4(rhs, np.zeros(33), 0.0, T, dt)
    late = t > 0.75 * T
    return np.sqrt((Z[late, :3] ** 2).sum(axis=1).mean())


def test_estimator_reduces_tracking_error():
    baseline = _toy_rms(0, 0, use_estimator=False)
    assert _toy_rms(cfg.ALPHA, cfg.BETA, True) < 0.7 * baseline  # paper rates: slow
    assert _toy_rms(cfg.ALPHA, 20.0, True) < 0.05 * baseline  # faster OFC learning
