import numpy as np
from numpy.polynomial import legendre as npleg

from sc_ie import basis


def test_legendre_matches_numpy():
    for x in np.linspace(-1, 1, 11):
        expected = [npleg.legval(x, np.eye(5)[i]) for i in range(5)]
        assert np.allclose(basis.legendre(x, 4), expected)


def test_legendre_matches_eq33_closed_forms():
    x = 0.37
    expected = [
        1.0,
        x,
        (3 * x**2 - 1) / 2,
        (5 * x**3 - 3 * x) / 2,
        (35 * x**4 - 30 * x**2 + 3) / 8,
    ]
    assert np.allclose(basis.legendre(x, 4), expected)


def test_legendre_regressor_is_block_diagonal_eq33():
    t, omega = 0.8, 2.0
    phi = basis.legendre_regressor(t, n=3, omega=omega)
    phi0 = basis.legendre(np.sin(omega * t))
    assert phi.shape == (3, 15)
    for i in range(3):
        assert np.allclose(phi[i, 5 * i : 5 * i + 5], phi0)
        assert np.count_nonzero(phi[i]) == np.count_nonzero(phi0)


def test_gaussian_eq7():
    g = basis.gaussian(0.5, centers=[0.5, 1.5], sigma=1.0)
    assert np.allclose(g, [1.0, np.exp(-1.0)])


def test_gaussian_regressor_uses_own_input_per_row():
    z = np.array([0.0, 1.0, -1.0])
    phi = basis.gaussian_regressor(z, centers=[-1.0, 0.0, 1.0], sigma=0.5)
    assert phi.shape == (3, 9)
    assert np.argmax(phi[0]) == 1  # z=0 peaks at centre 0
    assert np.argmax(phi[1]) == 5  # row 1 block starts at 3, centre +1
    assert np.argmax(phi[2]) == 6  # row 2 block starts at 6, centre -1
