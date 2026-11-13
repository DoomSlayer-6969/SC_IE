import numpy as np

from sc_ie.sync import mse_J, simulate_sync, steady_rms, sync_time


def test_mse_J_constant_error():
    t = np.linspace(0, 10, 1001)
    E = np.tile([1.0, 2.0, 2.0], (len(t), 1))
    assert np.isclose(mse_J(t, E), 9.0)


def test_sync_time_is_first_crossing():
    t = np.linspace(0, 1, 101)
    E = np.zeros((101, 3))
    E[:, 0] = np.exp(-10 * t)  # crosses 0.1 at t = ln(10)/10 ~ 0.23
    E[80:, 0] = 0.5  # later excursion must not move the sync time
    assert np.isclose(sync_time(t, E, tol=0.1), 0.24)
    assert np.isinf(sync_time(t, np.ones((101, 3)), tol=0.1))


def test_steady_rms():
    t = np.linspace(0, 2, 201)
    E = np.zeros((201, 3))
    E[t >= 1.0, 1] = 0.3
    assert np.isclose(steady_rms(t, E), 0.3)


def test_closed_loop_synchronizes_with_fast_poles():
    res = simulate_sync("legendre", poles=[-1000.0] * 3, T=1.0, dt=2e-4)
    assert res["t_sync"] < 0.05
    assert np.abs(res["E"][res["t"] > 0.1]).max() < 2.0
    assert np.abs(res["X"]).max() > 10  # drive actually on the chaotic attractor
