"""Phase 1 check: are the paper's systems chaotic with our parameter choices?

Run: python -m experiments.exp0_systems
Outputs: results/figures/exp0_systems.png, results/metrics/exp0_systems.json
"""

import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from sc_ie import config as cfg
from sc_ie import systems as S
from sc_ie.integrate import largest_lyapunov, rk4

DT = 1e-3  # coarser than cfg.DT; enough for attractor shape / LLE of the free systems
COLOR = "#2a78d6"


def chua(gamma):
    A = S.chua_A(gamma=gamma)
    return lambda t, x: S.chua_rhs(t, x, A)


# name -> (rhs, x0, LLE horizon [s], plotted state indices, label)
CASES = {
    "lu": (S.lu_rhs, cfg.LU_X0, 100.0, (0, 2), "Lu (Eq. 37)"),
    "genesio_tesi": (
        S.genesio_tesi_rhs,
        cfg.GT_Y0,
        1000.0,
        (0, 1),
        f"Genesio-Tesi (Eq. 38), a2,b2,c2={cfg.GT_A2},{cfg.GT_B2},{cfg.GT_C2}",
    ),
    "genesio_tesi_alt": (
        partial(S.genesio_tesi_rhs, a2=6.0, b2=2.92, c2=1.2),
        cfg.GT_Y0,
        1000.0,
        (0, 1),
        "Genesio-Tesi (Eq. 38), a2,b2,c2=6,2.92,1.2",
    ),
    "chua_gamma_0.385": (chua(0.385), cfg.CHUA_X0, 300.0, (0, 1), "Chua (Eq. 59), gamma=0.385"),
    "chua_gamma_0.0385": (chua(0.0385), cfg.CHUA_X0, 300.0, (0, 1), "Chua (Eq. 59), gamma=0.0385"),
}


def main():
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    metrics = {}
    for ax, (name, (f, x0, horizon, (i, j), label)) in zip(axes.flat, CASES.items()):
        lle = largest_lyapunov(f, x0, DT, horizon)
        _, X = rk4(f, np.array(x0, float), 0.0, 150.0, DT)
        X = X[len(X) // 3 :]  # drop transient
        metrics[name] = {"lle": round(lle, 4), "chaotic": bool(lle > 0.05)}
        print(f"{label:55s} LLE = {lle:+.4f}  -> {'chaotic' if lle > 0.05 else 'not chaotic'}")
        ax.plot(X[:, i], X[:, j], color=COLOR, lw=0.4)
        ax.set_title(f"{label}\nLLE = {lle:+.3f} 1/s", fontsize=9)
        ax.set_xlabel(f"state {i + 1}")
        ax.set_ylabel(f"state {j + 1}")

    # Eq. 40 response plant with u = 0 (dA as printed)
    ax = axes.flat[-1]
    t, Y = rk4(lambda t, y: S.response_rhs(t, y, np.zeros(3)), np.array(cfg.GT_Y0), 0, 10, DT)
    norm = np.linalg.norm(Y, axis=1)
    metrics["response_eq40_u0"] = {"norm_at_10s": float(norm[-1])}
    print(f"{'Response Eq. 40, u = 0':55s} |Y(10)| = {norm[-1]:.2e} (decays: dA[0,0] = -a1)")
    ax.semilogy(t, norm, color=COLOR, lw=1.5)
    ax.set_title(
        "Response plant Eq. 40 with u = 0\n(dA as printed -> decays to origin)", fontsize=9
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|Y(t)|")

    for ax in axes.flat:
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "exp0_systems.png", dpi=150)
    (cfg.METRICS_DIR / "exp0_systems.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
