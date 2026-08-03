from pathlib import Path


# ---------------------------------------------------------
# Project folders
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
GRAPHS_DIR = PROJECT_ROOT / "graphs"

TRAIN_DATA_DIR = PROJECT_ROOT / "traindata"
TEST_DATA_DIR = PROJECT_ROOT / "testdata"


# ---------------------------------------------------------
# Dataset files
# ---------------------------------------------------------

TRAIN_IMAGES_DIR = TRAIN_DATA_DIR / "IMG"
TRAIN_LOG_PATH = TRAIN_DATA_DIR / "driving_log.csv"

TEST_IMAGES_DIR = TEST_DATA_DIR / "IMG"
TEST_LOG_PATH = TEST_DATA_DIR / "driving_log.csv"


# ---------------------------------------------------------
# Saved outputs
# ---------------------------------------------------------

BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"
LOSS_GRAPH_PATH = GRAPHS_DIR / "training_loss.png"


# ---------------------------------------------------------
# Image settings
# ---------------------------------------------------------

IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3

INPUT_SHAPE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Number of consecutive poor validation epochs before stopping.
EARLY_STOPPING_PATIENCE = 3