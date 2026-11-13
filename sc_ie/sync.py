"""Closed-loop Lu -> response synchronization (Sections 3.3-3.4, 4.1).

State z = [X (3), Y (3), V (15), W (15)].
estimator: "legendre" (Eq. 33, phi of sin(omega t)), "gaussian" (Eq. 7, phi of E),
           or "none" (f_hat = 0, reference only).
"""

import numpy as np

from sc_ie import basis
from sc_ie import config as cfg
from sc_ie import controller as C
from sc_ie import systems as S
from sc_ie.integrate import rk4


def mse_J(t, E):
    """J = (1/T) integral_0^T E^T E dt (Eq. 58)."""
    return np.trapezoid((E**2).sum(axis=1), t) / (t[-1] - t[0])


def sync_time(t, E, tol=cfg.SYNC_TOL):
    """First time the initial transient brings ||E(t)|| below tol (inf if never)."""
    below = np.flatnonzero(np.linalg.norm(E, axis=1) < tol)
    return float(t[below[0]]) if below.size else float("inf")


def steady_rms(t, E, t_from=1.0):
    """RMS of ||E(t)|| after the transient (t >= t_from)."""
    return float(np.sqrt((E[t >= t_from] ** 2).sum(axis=1).mean()))


def simulate_sync(
    estimator="legendre",
    poles=cfg.SYNC_POLES,
    omega=cfg.OMEGA,
    alpha=cfg.ALPHA,
    beta=cfg.BETA,
    r=cfg.R,
    rho=cfg.RHO,
    q_scale=cfg.Q_SCALE,
    x0=cfg.LU_X0,
    y0=cfg.GT_Y0,
    T=cfg.T_END,
    dt=cfg.DT,
):
    A, B = S.lu_A(), np.eye(3)
    A_resp = A + S.delta_A()  # Eq. 40 as printed
    k = C.place(A, B, poles)
    P = C.riccati_P(A - B @ k, B, q_scale * np.eye(3), rho, r)
    nw = 3 * (cfg.LEGENDRE_ORDER + 1) if estimator == "legendre" else 3 * cfg.GAUSS_NUM

    def regressor(t, E):
        if estimator == "legendre":
            return basis.legendre_regressor(t, omega=omega)
        return basis.gaussian_regressor(E)

    def control(t, X, Y, V, W):
        E = Y - X
        s = C.s_signal(E, P, B)
        if estimator == "none":
            f_hat, dV, dW = np.zeros(3), np.zeros(nw), np.zeros(nw)
        else:
            phi = regressor(t, E)
            f_hat = C.estimate(phi, V, W)
            dV, dW = C.adaptation(phi, s, alpha, beta)
        u = C.sync_control(E, k, f_hat, C.robust_term(s, r))
        return u, f_hat, dV, dW

    def rhs(t, z):
        X, Y, V, W = z[:3], z[3:6], z[6 : 6 + nw], z[6 + nw :]
        u, _, dV, dW = control(t, X, Y, V, W)
        return np.concatenate([S.lu_rhs(t, X, A), S.response_rhs(t, Y, u, A_resp), dV, dW])

    z0 = np.concatenate([x0, y0, np.full(nw, cfg.V0), np.full(nw, cfg.W0)])
    t, Z = rk4(rhs, z0, 0.0, T, dt)
    X, Y = Z[:, :3], Z[:, 3:6]
    E = Y - X
    return {
        "t": t,
        "X": X,
        "Y": Y,
        "E": E,
        "Z": Z,
        "J": mse_J(t, E),
        "t_sync": sync_time(t, E),
        "rms_after_1s": steady_rms(t, E),
        "k": k,
        "P": P,
        "control": control,
        "nw": nw,
    }


def estimator_trace(res, stride=10):
    """Recompute f(t) (Eq. 42) and f_hat(t) along a stored run, every `stride` samples."""
    nw, idx = res["nw"], np.arange(0, len(res["t"]), stride)
    f_true, f_hat = [], []
    for i in idx:
        z = res["Z"][i]
        X, Y, V, W = z[:3], z[3:6], z[6 : 6 + nw], z[6 + nw :]
        f_true.append(S.sync_uncertainty(X, Y))
        f_hat.append(res["control"](res["t"][i], X, Y, V, W)[1])
    return res["t"][idx], np.array(f_true), np.array(f_hat)
