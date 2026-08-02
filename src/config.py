from pathlib import Path


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
GRAPHS_DIR = PROJECT_ROOT / "graphs"
DATASET_DIR = PROJECT_ROOT / "dataset"


# Dataset files
IMAGES_DIR = DATASET_DIR / "IMG"
DRIVING_LOG_PATH = DATASET_DIR / "driving_log.csv"


# Model files
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"


# Training graph
LOSS_GRAPH_PATH = GRAPHS_DIR / "training_loss.png"


# Image settings required by the Nvidia model
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


# Training settings
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
RANDOM_SEED = 42