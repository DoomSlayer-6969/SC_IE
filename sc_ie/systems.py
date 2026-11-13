"""Chaotic systems of the paper.

Drive  : Lu system, Eq. 37 / 39            X' = A X + g(X)
Response: Eq. 40 exactly as printed         Y' = (A + dA) Y + g(Y) + B u,  B = I3
Comm.  : Chua transmitter, Eq. 59 / 43 / 49
"""

import numpy as np

from sc_ie import config as cfg

# ---------------------------------------------------------------------------
# Lu drive system (Eq. 37 / 39)
# ---------------------------------------------------------------------------


def lu_A(a1=cfg.LU_A1, b1=cfg.LU_B1, c1=cfg.LU_C1):
    """Linear part A of the Lu system (Eq. 39)."""
    return np.array([[-a1, a1, 0.0], [0.0, b1, 0.0], [0.0, 0.0, -c1]])


def lu_g(x):
    """Nonlinear part g(X) of the Lu system (Eq. 39)."""
    return np.array([0.0, -x[0] * x[2], x[0] * x[1]])


def lu_rhs(t, x, A=None):
    A = lu_A() if A is None else A
    return A @ x + lu_g(x)


# ---------------------------------------------------------------------------
# Genesio-Tesi response system (Eq. 38 / 40)
# ---------------------------------------------------------------------------


def delta_A(a1=cfg.LU_A1, b1=cfg.LU_B1, c1=cfg.LU_C1, a2=cfg.GT_A2, b2=cfg.GT_B2, c2=cfg.GT_C2):
    """dA exactly as printed in Eq. 40 (note dA[0,0] = -a1, so A + dA != Eq. 38)."""
    return np.array(
        [
            [-a1, 1.0 - a1, 0.0],
            [0.0, -b1, 1.0],
            [-a2, -b2, -(c2 - c1)],
        ]
    )


def gt_g(y):
    """Nonlinear part g(Y) of the response system (Eq. 40)."""
    return np.array([0.0, 0.0, y[0] ** 2])


def response_rhs(t, y, u, A_resp=None):
    """Response plant of Eq. 40: Y' = (A + dA) Y + g(Y) + u, with B = I3."""
    A_resp = lu_A() + delta_A() if A_resp is None else A_resp
    return A_resp @ y + gt_g(y) + u


def genesio_tesi_rhs(t, y, a2=cfg.GT_A2, b2=cfg.GT_B2, c2=cfg.GT_C2):
    """Uncontrolled Genesio-Tesi system as written in Eq. 38 (reference only)."""
    return np.array([y[1], y[2], -a2 * y[0] - b2 * y[1] - c2 * y[2] + y[0] ** 2])


def sync_uncertainty(x, y, dA=None):
    """f(t) = dA Y + g(Y) - g(X) from Eq. 42, so that E' = A E + f(t) + B u."""
    dA = delta_A() if dA is None else dA
    return dA @ y + gt_g(y) - lu_g(x)


# ---------------------------------------------------------------------------
# Chua transmitter (Eq. 59), written as X' = A X + g(y) + B d(t)
# ---------------------------------------------------------------------------

CHUA_B = np.array([1.0, 0.0, 0.0])  # d(t) enters the first equation (Eq. 59)
CHUA_C = np.array([1.0, 0.0, 0.0])  # y = [1 0 0] X = x1 (Eq. 59)


def chua_A(p=cfg.CHUA_P, q=cfg.CHUA_Q, gamma=cfg.CHUA_GAMMA):
    """Linear part of Eq. 59 (the p*g(x1) term is kept in the nonlinearity)."""
    return np.array([[-p, p, 0.0], [1.0, -1.0, 1.0], [0.0, -q, -gamma]])


def chua_h(x1, a=cfg.CHUA_A, b=cfg.CHUA_B):
    """Piecewise-linear Chua diode, g(x1) in Eq. 59."""
    return b * x1 + 0.5 * (a - b) * (np.abs(x1 + 1.0) - np.abs(x1 - 1.0))


def chua_g(y, p=cfg.CHUA_P, a=cfg.CHUA_A, b=cfg.CHUA_B):
    """Vector nonlinearity g(y) of Eq. 43/49, a function of the scalar output y only."""
    return np.array([-p * chua_h(y, a, b), 0.0, 0.0])


def chua_d(t):
    """External disturbance d(t) = cos(t) (Eq. 59)."""
    return np.cos(t)


def chua_rhs(t, x, A=None, with_disturbance=True):
    """Autonomous Chua transmitter of Eq. 59 (no message injected)."""
    A = chua_A() if A is None else A
    dx = A @ x + chua_g(x[0])
    if with_disturbance:
        dx = dx + CHUA_B * chua_d(t)
    return dx
