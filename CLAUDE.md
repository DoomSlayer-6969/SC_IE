
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: SC_IE paper replication

Python replication of `SC_IE.pdf`: Rahiman, Majidi & Khorashadizadeh (2026), *Secure communication and image encryption using emotional learning-based chaos synchronization*, MTAP 85:680. Cite equations by their paper number (Eq. 40, Fig. 12, ...) in code and docs.

## Commands

All tools run from the repo root through `.venv` (Python 3.14).

```
make install-dev                      # deps + pytest/ruff
make check                            # ruff check + full pytest (~30 s)
make format                           # ruff format
.venv/bin/python -m pytest tests/test_systems.py::test_lu_matches_eq37 -q   # single test
.venv/bin/python -m experiments.exp1_sync                                   # run one experiment
```

- Run experiments as modules (`-m experiments.<name>`) from the repo root. `pyproject.toml` puts `.` on pytest's path, but ad-hoc scripts need `PYTHONPATH=.`.
- `exp1_sync` takes about 6 minutes (many 10 s simulations at dt = 1e-4).

## Ground rules for this repo

- **Implement equations exactly as printed, even suspected typos.** The user's standing instruction (e.g. keep `ΔA[0,0] = −a1` in Eq. 40). Document any inconsistency, with its consequence, in the code docstring and in `results/REPORT.md`. Never silently correct it or tune values to match the paper's numbers.
- **Tag every parameter in `sc_ie/config.py`** as `[paper]` (with the equation number) or `[assumed]`. Parameters the paper doesn't give live only there.
- **Work in approved phases.** Finish a phase, report what was verified, then ask before starting the next one. Never commit; the user commits.
- `data/input_images/peppers.png` is supplied by the user. Don't download a substitute.

## Architecture

`sc_ie/` holds pure library functions. `experiments/` scripts wire them into closed loops and write `results/figures/*.png` and `results/metrics/*.json`. Importing `config` has no side effects; experiments create output directories themselves.

- `integrate.py`: fixed-step RK4. It's deliberate: the control and cipher laws contain `max`, `mod`, `|·|` and piecewise terms, so adaptive solvers don't help. It also has a Benettin largest-Lyapunov estimate, used to check that a system is chaotic.
- `systems.py`: Lü drive (Eq. 37/39), response plant `Y' = (A+ΔA)Y + g(Y) + u` from **Eq. 40 as printed** (this differs from Eq. 38 at `[0,0]`; a test pins this down), the Eq. 42 uncertainty `f = ΔA·Y + g(Y) − g(X)`, and the Chua transmitter (Eq. 59) with `B = C = [1,0,0]`.
- `basis.py`: Legendre P0..P4 of `sin(ωt)` as a block-diagonal 3×15 regressor (Eq. 33). The Gaussian baseline (Eq. 7) takes the sync error `E` as its input.
- `controller.py`: pole placement, the Eq. 20 Riccati equation solved in matrix form via CARE, `S = EᵀPB`, `f̂ = φ(V−W)`, the Eq. 35–36 adaptation laws, `u_r` (Eq. 19) and `u` (Eq. 32). Its module docstring records two interpretation choices:
  - `P` is applied once, through `S`, because the printed form doesn't have consistent dimensions for 3 states;
  - Eq. 19's sign is kept as printed, which makes `u_r` positive feedback.
- `sync.py`: closed-loop simulation with state `z = [X, Y, V, W]`; `estimator` is one of `legendre | gaussian | none`.

## Experiments and key findings so far

| Script | Paper result | Notes |
|---|---|---|
| `exp0_systems` | chaos checks | Genesio–Tesi (6, 2.92, 1.2) is periodic, so config uses (1, 1.1, 0.44). Chua `γ = 0.385` (as printed) is not chaotic; `0.0385` is. The user chose to run 0.385 by default with 0.0385 as a variant. |
| `exp1_sync` | §4.1, Figs 6–11, J | With the paper's `α = 0.01`, `β = 1`, the estimator contributes nothing (effective learning rate `β·P` ≈ 4e-4). `J ∝ 1/pole²`. `SYNC_POLES = −1380.5` reproduces J = 0.1661, but Gaussian and no-estimator give the same J at equal gains, not the paper's 0.226. |

Planned (approved plan):
- `exp2_secure_comm`: Chua observer, Figs 12–14. Oracle vs output-only adaptation, both `γ` values.
- `exp3_encrypted_comm`: n-shift cipher, Fig. 15.
- `exp4_image`: XOR keystream from Eq. 61, Figs 16–21, Tables 2–5.
- `results/REPORT.md`: paper vs replicated numbers.

Known paper issues to report rather than fix: NPCR/UACI and the noise-attack PSNR can't come from a plaintext-independent XOR keystream; the n-shift cipher with n = 30 collapses to a single shift by `nK`; the "6%" MSE claim doesn't match the paper's own numbers.
