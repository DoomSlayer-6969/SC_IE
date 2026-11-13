import numpy as np

from sc_ie.integrate import largest_lyapunov, rk4


def test_rk4_exponential_decay():
    t, x = rk4(lambda t, x: -x, np.array([1.0]), 0.0, 1.0, 1e-3)
    assert np.max(np.abs(x[:, 0] - np.exp(-t))) < 1e-10


def test_rk4_non_autonomous():
    t, x = rk4(lambda t, x: np.array([np.cos(t)]), np.array([0.0]), 0.0, 5.0, 1e-3)
    assert np.max(np.abs(x[:, 0] - np.sin(t))) < 1e-10


def test_rk4_fourth_order():
    def err(dt):
        _, x = rk4(lambda t, x: -x, np.array([1.0]), 0.0, 1.0, dt)
        return abs(x[-1, 0] - np.exp(-1.0))

    ratio = err(0.1) / err(0.05)
    assert 14 < ratio < 18  # 2^4 = 16


def test_lyapunov_of_stable_linear_system():
    lle = largest_lyapunov(lambda t, x: -2.0 * x, [1.0, 1.0], 1e-3, 5.0, t_transient=0.0)
    assert abs(lle + 2.0) < 1e-3
