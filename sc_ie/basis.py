"""Thalamus basis functions: Legendre polynomials (Eq. 28, 33) and Gaussian RBFs (Eq. 7)."""

import numpy as np

from sc_ie import config as cfg


def legendre(x, order=cfg.LEGENDRE_ORDER):
    """[P0(x), ..., P_order(x)] via the recurrence of Eq. 28."""
    p = np.empty(order + 1)
    p[0] = 1.0
    if order >= 1:
        p[1] = x
    for i in range(1, order):
        p[i + 1] = ((2 * i + 1) * x * p[i] - i * p[i - 1]) / (i + 1)
    return p


def legendre_regressor(t, n=3, omega=cfg.OMEGA, order=cfg.LEGENDRE_ORDER):
    """phi of Eq. 33: n x n(order+1) block-diagonal, phi0 = legendre(sin(omega t)) per row."""
    return np.kron(np.eye(n), legendre(np.sin(omega * t), order))


def gaussian(z, centers=cfg.GAUSS_CENTERS, sigma=cfg.GAUSS_SIGMA):
    """Gaussian RBFs of Eq. 7 for a scalar input z: exp(-(z - mu_j)^2 / sigma^2)."""
    return np.exp(-((z - np.asarray(centers)) ** 2) / sigma**2)


def gaussian_regressor(z, centers=cfg.GAUSS_CENTERS, sigma=cfg.GAUSS_SIGMA):
    """Block-diagonal phi whose row i holds the Gaussian RBFs of input z[i]."""
    n, m = len(z), len(centers)
    phi = np.zeros((n, n * m))
    for i, zi in enumerate(z):
        phi[i, i * m : (i + 1) * m] = gaussian(zi, centers, sigma)
    return phi
