from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from config import GRAPHS_DIR, LOSS_GRAPH_PATH, MODELS_DIR


def create_output_directories() -> None:
    """
    Create the folders used for saved models and training graphs.
    """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def save_training_history(
    history: Dict[str, List[float]],
    output_path: Path = LOSS_GRAPH_PATH,
) -> None:
    """
    Save the training and validation loss graph.

    Parameters
    ----------
    history:
        Dictionary containing the training history values.

    output_path:
        Location where the graph image will be saved.
    """

    if "loss" not in history:
        raise ValueError("Training history does not contain a 'loss' value.")

    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"], label="Training Loss")

    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")

    plt.title("Model Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    create_output_directories()

    test_history = {
        "loss": [0.30, 0.20, 0.14, 0.10],
        "val_loss": [0.34, 0.24, 0.18, 0.15],
    }

    save_training_history(test_history)

    print("Output folders verified.")
    print("Test graph saved to:", LOSS_GRAPH_PATH)