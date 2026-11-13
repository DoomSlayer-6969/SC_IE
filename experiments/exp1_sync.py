"""Phase 4 / Section 4.1: Lu -> response synchronization, Legendre vs Gaussian estimator.

Run: python -m experiments.exp1_sync
Outputs:
  results/figures/exp1_sync_{legendre,gaussian}.png   (Figs 9-11 / 6-8)
  results/figures/exp1_error_and_estimate.png         ||E(t)|| and f vs f_hat
  results/figures/exp1_pole_sweep.png                 J vs pole speed
  results/metrics/exp1_sync.json
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sc_ie import config as cfg
from sc_ie.sync import estimator_trace, simulate_sync

PAPER_J = {"legendre": 0.1661, "gaussian": 0.226}
ESTIMATORS = ["legendre", "gaussian", "none"]
COLORS = {"legendre": "#2a78d6", "gaussian": "#eb6834", "none": "#1baf7a"}
LABELS = {
    "legendre": "Legendre (proposed)",
    "gaussian": "Gaussian (baseline)",
    "none": "no estimator",
}
SWEEP_DT = 2e-4  # coarser step for sweeps; final runs use cfg.DT


def run(estimator, pole, dt=SWEEP_DT, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # divergent runs overflow; reported as inf
        res = simulate_sync(estimator, poles=[pole] * 3, dt=dt, **kw)
    if not np.isfinite(res["J"]):
        res["J"] = float("inf")
    return res


def calibrate_pole(target, pole=-1000.0, iters=3):
    """Pole (repeated x3) at which the Legendre run gives J = target; uses J ~ 1/pole^2."""
    for _ in range(iters):
        J = run("legendre", pole)["J"]
        pole *= np.sqrt(J / target)
    return float(pole)


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_states(res, name, path):
    t, X, Y = res["t"], res["X"], res["Y"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, X[:, i], color="#2a78d6", lw=1.2, label=f"x{i + 1} (Lu drive)")
        ax.plot(t, Y[:, i], color="#eb6834", lw=1.2, ls="--", label=f"y{i + 1} (response)")
        ax.set_ylabel("state variables")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        style(ax)
        ins = ax.inset_axes([0.33, 0.52, 0.32, 0.45])
        zoom = t <= 0.6
        ins.plot(t[zoom], X[zoom, i], color="#2a78d6", lw=1)
        ins.plot(t[zoom], Y[zoom, i], color="#eb6834", lw=1, ls="--")
        ins.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"{LABELS[name]}: J = {res['J']:.4f}, t_sync = {res['t_sync']:.3f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {"paper": PAPER_J}

    # 1. J vs pole speed for all estimators (paper alpha, beta, r; Q = Q_SCALE * I)
    poles = [-50.0, -100.0, -200.0, -500.0, -1000.0, -2000.0]
    sweep = {e: [run(e, p)["J"] for p in poles] for e in ESTIMATORS}
    metrics["pole_sweep"] = {"poles": poles, **sweep}
    print("J vs pole:", json.dumps(metrics["pole_sweep"], indent=1))

    # 2. Calibrate the pole so Legendre reproduces the paper's J, then compare at equal gains
    pole = calibrate_pole(PAPER_J["legendre"])
    metrics["calibrated_pole"] = pole
    print(f"calibrated pole = {pole:.1f}")

    # 3. omega sweep for Legendre at the calibrated pole
    omegas = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    metrics["omega_sweep"] = {
        "omegas": omegas,
        "J": [run("legendre", pole, omega=w)["J"] for w in omegas],
    }

    # 4. Diagnostic beyond the paper: larger OFC rate beta at slow poles
    betas = [1.0, 1e2, 1e4]
    metrics["beta_diagnostic_pole_-50"] = {
        "betas": betas,
        **{e: [run(e, -50.0, beta=b)["J"] for b in betas] for e in ["legendre", "gaussian"]},
        "none": run("none", -50.0)["J"],
    }

    # 5. Final runs at cfg.DT and the calibrated pole
    final = {e: run(e, pole, dt=cfg.DT) for e in ESTIMATORS}
    metrics["final"] = {
        e: {
            "J": r["J"],
            "t_sync": r["t_sync"],
            "rms_after_1s": r["rms_after_1s"],
            "P11": float(r["P"][0, 0]),
        }
        for e, r in final.items()
    }
    J_l, J_g = final["legendre"]["J"], final["gaussian"]["J"]
    metrics["final"]["reduction_pct"] = 100 * (J_g - J_l) / J_g
    metrics["paper_reduction_pct"] = (
        100 * (PAPER_J["gaussian"] - PAPER_J["legendre"]) / PAPER_J["gaussian"]
    )
    for e, r in final.items():
        print(
            f"{e:9s} J = {r['J']:.5f}  t_sync = {r['t_sync']:.4f} s  rms(t>1s) = {r['rms_after_1s']:.3f}"
        )
    print(
        f"Legendre vs Gaussian reduction: {metrics['final']['reduction_pct']:.2f}% "
        f"(paper: {metrics['paper_reduction_pct']:.1f}%, text claims 6%)"
    )

    plot_states(final["legendre"], "legendre", cfg.FIGURES_DIR / "exp1_sync_legendre.png")
    plot_states(final["gaussian"], "gaussian", cfg.FIGURES_DIR / "exp1_sync_gaussian.png")

    # ||E|| over time and f vs f_hat (state 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    for e, r in final.items():
        ax1.semilogy(
            r["t"],
            np.linalg.norm(r["E"], axis=1) + 1e-12,
            color=COLORS[e],
            lw=1,
            label=f"{LABELS[e]} (J = {r['J']:.4f})",
        )
    ax1.axhline(cfg.SYNC_TOL, color="#888", lw=0.8, ls=":")
    ax1.set_ylabel("||E(t)||")
    ax1.set_xlabel("Time (s)")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(f"Synchronization error, poles = {pole:.0f} (x3)")
    tt, f_true, f_hat = estimator_trace(final["legendre"])
    ax2.plot(tt, f_true[:, 0], color="#2a78d6", lw=1, label="f1(t) true (Eq. 42)")
    ax2.plot(tt, f_hat[:, 0], color="#eb6834", lw=1.5, label="f1_hat(t) Legendre estimate")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("uncertainty, state 1")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        style(ax)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "exp1_error_and_estimate.png", dpi=150)
    plt.close(fig)

    # J vs pole speed
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in ESTIMATORS:
        ax.loglog(
            [-p for p in poles],
            sweep[e],
            color=COLORS[e],
            lw=2,
            marker="o",
            ms=8,
            ls="-" if e != "none" else ":",
            label=LABELS[e],
        )
    for e, y in PAPER_J.items():
        ax.axhline(y, color=COLORS[e], lw=0.8, ls="--")
        ax.text(55, y * 1.08, f"paper J ({e}) = {y}", fontsize=8, color="#444")
    ax.set_xlabel("|pole| of A - Bk")
    ax.set_ylabel("J (Eq. 58)")
    ax.set_title("J vs feedback gain (alpha = 0.01, beta = 1, r = 0.08)")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "exp1_pole_sweep.png", dpi=150)
    plt.close(fig)

    (cfg.METRICS_DIR / "exp1_sync.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
