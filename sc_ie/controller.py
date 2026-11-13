"""Emotional-learning (BELBIC / CRBENN) controller pieces, Sections 3.2-3.4.

Error dynamics (Eq. 42):  E' = A E + f(t) + B u
Control law  (Eq. 32):    u = -k E - f_hat - u_r
Estimator    (Eq. 34):    f_hat = (V - W)^T phi           (here: phi @ (V - W))
Adaptation   (Eq. 35-36): V' = alpha phi^T max(s, 0),  W' = -beta phi^T s
Robust term  (Eq. 19):    u_r = -(1/r) s
Signal s     (Sec. 4.1):  S = E^T P B

Reading of P: Eqs. 19, 23-24 come from a scalar derivation in which s is the raw
error, so P multiplies it there. Section 4.1 defines the MIMO signal as
S = E^T P B, which already contains P, so P is applied once (inside S) everywhere.
With B = I3 and g(x) = 1 this is the dimensionally consistent form of the printed
equations. The sign of Eq. 19 is kept as printed (see results/REPORT.md).
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from scipy.signal import place_poles

from sc_ie import config as cfg


def place(A, B, poles):
    """Gain k such that eig(A - B k) = poles (pole placement, Section 3.3)."""
    return place_poles(A, B, poles).gain_matrix


def riccati_P(A_c, B, Q, rho=cfg.RHO, r=cfg.R):
    """P = P^T > 0 solving the matrix form of Eq. 20 with -2K -> A_c:

    A_c^T P + P A_c + Q + P B (1/rho^2 - 2/r) B^T P = 0,  requires 2 rho^2 >= r (Eq. 21).
    """
    c = 2.0 / r - 1.0 / rho**2
    if c < 0:
        raise ValueError("Eq. 21 violated: need 2 rho^2 >= r")
    if c == 0:
        return solve_continuous_lyapunov(A_c.T, -Q)
    return solve_continuous_are(A_c, B, Q, np.eye(B.shape[1]) / c)


def s_signal(E, P, B):
    """S = E^T P B (Section 4.1), returned as a 1-D array of length B.shape[1]."""
    return B.T @ P @ E


def estimate(phi, V, W):
    """Emotional estimate f_hat = (V - W)^T phi (Eq. 34)."""
    return phi @ (V - W)


def adaptation(phi, s, alpha=cfg.ALPHA, beta=cfg.BETA):
    """Amygdala / OFC weight rates (Eq. 35-36) with g(x) = 1."""
    return alpha * phi.T @ np.maximum(s, 0.0), -beta * phi.T @ s


def robust_term(s, r=cfg.R):
    """Robust term u_r = -(1/r) s (Eq. 19, sign as printed)."""
    return -s / r


def sync_control(E, k, f_hat, u_r):
    """Synchronization control u = -k E - f_hat - u_r (Eq. 32)."""
    return -k @ E - f_hat - u_r
