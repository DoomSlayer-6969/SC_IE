"""Fixed-step RK4 integration and a largest-Lyapunov-exponent estimate."""

import numpy as np


def rk4_step(f, t, x, dt):
    """One classical RK4 step for x' = f(t, x)."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(f, x0, t0, t1, dt):
    """Integrate x' = f(t, x) from t0 to t1. Returns (t, X) with X[i] = x(t[i])."""
    n = round((t1 - t0) / dt)
    t = t0 + dt * np.arange(n + 1)
    x = np.empty((n + 1, len(x0)))
    x[0] = x0
    for i in range(n):
        x[i + 1] = rk4_step(f, t[i], x[i], dt)
    return t, x


def largest_lyapunov(f, x0, dt, t_total, t_transient=20.0, d0=1e-8, renorm_every=10):
    """Benettin two-trajectory estimate of the largest Lyapunov exponent (1/s).

    Returns nan if the trajectory leaves a bounded region (|x| > 1e6).
    """
    x = np.asarray(x0, dtype=float)
    t = 0.0
    for _ in range(round(t_transient / dt)):
        x = rk4_step(f, t, x, dt)
        t += dt
    xp = x + d0 / np.sqrt(len(x))
    log_sum = 0.0
    steps = round(t_total / dt)
    for i in range(1, steps + 1):
        x = rk4_step(f, t, x, dt)
        xp = rk4_step(f, t, xp, dt)
        t += dt
        if i % renorm_every == 0:
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
                return float("nan")
            d = np.linalg.norm(xp - x)
            log_sum += np.log(d / d0)
            xp = x + (xp - x) * (d0 / d)
    return log_sum / (steps * dt)
