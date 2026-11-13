"""
Central configuration for the SC_IE replication of:
  Rahiman, Majidi & Khorashadizadeh (2026), "Secure communication and image
  encryption using emotional learning-based chaos synchronization",
  Multimedia Tools and Applications 85:680.

Every value is tagged:
  [paper]   taken from the paper (equation / section cited)
  [assumed] not given in the paper; chosen by us and listed in results/REPORT.md
"""

from pathlib import Path

# =====================================================
# 1. PROJECT DIRECTORIES
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
INPUT_IMAGE_DIR = DATA_DIR / "input_images"
IMAGE_PATH = INPUT_IMAGE_DIR / "peppers.png"  # supplied by the user

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"


# =====================================================
# 2. SIMULATION SETTINGS
# =====================================================

T_START = 0.0
T_END = 10.0  # [paper] Eq. 58 and Figs 6-15 span 0..10 s

# [assumed] fixed-step RK4; the control/cipher laws contain max(), mod and
# piecewise terms, so an adaptive solver gives no accuracy benefit.
DT = 1e-4


# =====================================================
# 3. DRIVE SYSTEM: LU (Eq. 37 / 39)
#   x1' = a1 (x2 - x1)
#   x2' = -x1 x3 + b1 x2
#   x3' = x1 x2 - c1 x3
# =====================================================

LU_A1 = 36.0  # [assumed] standard Lu & Chen (2002), ref [49]
LU_B1 = 20.0  # [assumed]
LU_C1 = 3.0  # [assumed]

LU_X0 = [1.0, 1.0, 1.0]  # [assumed]


# =====================================================
# 4. RESPONSE SYSTEM: GENESIO-TESI (Eq. 38 / 40)
#   Simulated as Y' = (A + dA) Y + g(Y) + B u with dA exactly as printed
#   in Eq. 40 (including dA[0,0] = -a1, which differs from Eq. 38).
# =====================================================

# [assumed] Genesio & Tesi (1992), ref [50], chaotic set (a,b,c) = (0.44, 1.1, 1).
# The other common set (1.2, 2.92, 6) settles on a periodic orbit (Phase 1 check).
GT_A2 = 1.0  # coefficient of y1 in Eq. 38
GT_B2 = 1.1  # coefficient of y2
GT_C2 = 0.44  # coefficient of y3

GT_Y0 = [0.1, 0.1, 0.1]  # [assumed] inside the Genesio-Tesi basin of attraction


# =====================================================
# 5. EMOTIONAL LEARNING CONTROLLER (Sections 3.2-3.3, 4)
# =====================================================

ALPHA = 0.01  # [paper] Section 4, Amygdala learning rate
BETA = 1.0  # [paper] Section 4, OFC learning rate
R = 0.08  # [paper] Section 4, robust-term constant r

RHO = 0.5  # [assumed] must satisfy 2*rho^2 >= r (Eq. 21)
Q_SCALE = 1.0  # [assumed] Q = Q_SCALE * I in the Riccati equation (Eq. 20)

# [assumed] eigenvalues of A - B k. Calibrated in Phase 4 (experiments/exp1_sync.py) so the
# Legendre run reproduces the paper's J = 0.1661; slower poles do not synchronize (J ~ 1/pole^2).
SYNC_POLES = [-1380.5, -1380.5, -1380.5]

SYNC_TOL = 0.1  # [assumed] sync time = first time ||E|| < SYNC_TOL

V0 = 0.0  # [assumed] initial Amygdala weights
W0 = 0.0  # [assumed] initial OFC weights


# =====================================================
# 6. BASIS FUNCTIONS
# =====================================================

# Legendre (Eq. 28, 33): P0..P4 of x = sin(omega t)
LEGENDRE_ORDER = 4  # [paper] Eq. 33
OMEGA = 1.0  # [assumed] not given; swept in Phase 4

# Gaussian baseline (Eq. 7, Section 4.1)
GAUSS_NUM = 5  # [paper] remark after Eq. 36 (m = 5)
GAUSS_CENTERS = [-1.0, -0.5, 0.0, 0.5, 1.0]  # [assumed]
GAUSS_SIGMA = 0.5  # [assumed]


# =====================================================
# 7. SECURE COMMUNICATION: CHUA TRANSMITTER (Eq. 59)
#   x1' = p (x2 - x1 - g(x1)) + d(t)
#   x2' = x1 - x2 + x3
#   x3' = -q x2 - gamma x3
#   g(x1) = b x1 + 0.5 (a - b)(|x1 + 1| - |x1 - 1|),  d(t) = cos(t)
# =====================================================

CHUA_P = 10.0  # [paper] Eq. 59
CHUA_Q = 15.0  # [paper] Eq. 59
CHUA_GAMMA = 0.385  # [paper] Eq. 59 (possibly a typo for 0.0385; checked in Phase 1)
CHUA_A = -1.27  # [assumed] standard Chua slope
CHUA_B = -0.68  # [assumed] standard Chua slope

CHUA_X0 = [0.1, 0.0, 0.0]  # [assumed]
OBSERVER_X0 = [0.0, 0.0, 0.0]  # [assumed]
OBSERVER_POLES = [-5.0, -6.0, -7.0]  # [assumed] eigenvalues of A - L C

# "oracle": adaptation driven by the true state error (paper fidelity)
# "output": adaptation driven only by the received signal (realizable variant)
OBSERVER_ADAPTATION = "oracle"


# =====================================================
# 8. MESSAGES AND N-SHIFT CIPHER (Eqs. 50-55, 60)
# =====================================================

MESSAGE_FREQ = 20.0  # [paper] Eq. 60, m(t) = sin(20 t)
# [paper] Section 4.2 / Fig. 13: (t_start, t_end, rad/s)
MULTI_FREQ_SEGMENTS = [(0.0, 3.0, 1.0), (3.0, 7.0, 10.0), (7.0, 10.0, 50.0)]
SQUARE_FREQ = 20.0  # [assumed] Fig. 14 ~ sign(sin(20 t))

N_SHIFT = 30  # [paper] Section 4.2, n
H_SHIFT = 2.0  # [paper] Section 4.2, h
KEY_STATE = 1  # [paper] Eq. 50, K(t) = x2(t) (0-based index)


# =====================================================
# 9. IMAGE ENCRYPTION (Eq. 61)
# =====================================================

IMAGE_SIZE = 256  # [paper] Section 5, 256 x 256 grayscale
KEY_A = 125_000_000  # [paper] Eq. 61, a = 125e6
KEY_B = 539  # [paper] Eq. 61, b

KEYSTREAM_STATE = 1  # [assumed] synchronized x2-hat feeds the keystream
KEYSTREAM_DISCARD = 1.0  # [assumed] seconds of transient skipped before sampling


# =====================================================
# 10. SECURITY ANALYSIS (Section 5)
# =====================================================

NOISE_DENSITIES = [0.005, 0.05, 0.1]  # [paper] Section 5.5
RANDOM_SEED = 42  # [assumed]
