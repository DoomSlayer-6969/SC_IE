"""
Central configuration file
Project: Secure Communication and Image Encryption
Using Emotional Learning-Based Chaos Synchronization
"""

from pathlib import Path

# =====================================================
# 1. PROJECT DIRECTORIES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
INPUT_IMAGE_DIR = DATA_DIR / "input_images"
OUTPUT_IMAGE_DIR = DATA_DIR / "output_images"

RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"


# =====================================================
# 2. SIMULATION SETTINGS
# =====================================================

T_START = 0.0
T_END = 10.0

# Numerical integration settings
# Implementation choice, not specified by the paper
TIME_STEP = 0.001

SOLVER = "RK45"
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 1e-9


# =====================================================
# 3. LU CHAOTIC SYSTEM PARAMETERS
# =====================================================

# Values must be verified from the paper/reference model
LU_A1 = None
LU_B1 = None
LU_C1 = None

LU_INITIAL_STATE = [1.0, 1.0, 1.0]


# =====================================================
# 4. GENESIO-TESI CHAOTIC SYSTEM PARAMETERS
# =====================================================

GENESIO_A2 = None
GENESIO_B2 = None
GENESIO_C2 = None

GENESIO_INITIAL_STATE = [1.0, 1.0, 1.0]


# =====================================================
# 5. LEGENDRE POLYNOMIAL SETTINGS
# =====================================================

LEGENDRE_ORDER = 4
LEGENDRE_NUM_FUNCTIONS = LEGENDRE_ORDER + 1

# x = sin(omega * t)
OMEGA = None


# =====================================================
# 6. BELBIC CONTROLLER PARAMETERS
# =====================================================

ALPHA = 0.01       # Amygdala learning rate
BETA = 1.0         # OFC learning rate

ROBUST_R = 0.08

# Initial adaptive weights
INITIAL_V = 0.0
INITIAL_W = 0.0


# =====================================================
# 7. SYNCHRONIZATION SETTINGS
# =====================================================

NUM_STATES = 3

# Controller gains and observer poles must be
# determined from the paper's design procedure.
STATE_FEEDBACK_GAIN = None
OBSERVER_POLES = None

INITIAL_SYNCHRONIZATION_ERROR = [
    1.0, 1.0, 1.0
]


# =====================================================
# 8. COMMUNICATION SETTINGS
# =====================================================

MESSAGE_FREQUENCY = 20.0
MESSAGE_TYPE = "sine"

ENCRYPTION_BLOCKS = 30
SHIFT_PARAMETER_H = 2.0

# Communication simulation uses a separate
# system described in Section 4.2.
COMMUNICATION_A = None
COMMUNICATION_B = None


# =====================================================
# 9. IMAGE ENCRYPTION SETTINGS
# =====================================================

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

IMAGE_KEY_A = 125 * 10**6
IMAGE_KEY_B = 539

PIXEL_MODULUS = 256
PIXEL_DTYPE = "uint8"


# =====================================================
# 10. SECURITY ANALYSIS SETTINGS
# =====================================================

NOISE_DENSITIES = [0.005, 0.05, 0.1]

ENTROPY_MAX = 8.0

CORRELATION_DIRECTIONS = [
    "horizontal",
    "vertical",
    "diagonal"
]


# =====================================================
# 11. RANDOMNESS / REPRODUCIBILITY
# =====================================================

RANDOM_SEED = 42


# =====================================================
# 12. CREATE REQUIRED DIRECTORIES
# =====================================================

for directory in [
    INPUT_IMAGE_DIR,
    OUTPUT_IMAGE_DIR,
    GRAPHS_DIR,
    METRICS_DIR,
    LOGS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
