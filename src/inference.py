from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


def load_steering_model(model_path: Path):
    """
    Load a saved steering-angle model.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    return load_model(
    model_path,
    safe_mode=False,
    compile=False,
)


def predict_steering(model, image: np.ndarray) -> float:
    """
    Predict one steering angle from a preprocessed image.

    Expected image shape:
        (66, 200, 3)
    """

    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    )

    if image.shape != expected_shape:
        raise ValueError(
            f"Expected image shape {expected_shape}, "
            f"but received {image.shape}."
        )

    image_batch = np.expand_dims(image, axis=0)

    prediction = model.predict(
        image_batch,
        verbose=0,
    )

    return float(prediction[0][0])


if __name__ == "__main__":
    model_path = Path("models/untrained_nvidia_model.keras")

    steering_model = load_steering_model(model_path)

    dummy_image = np.random.randint(
        0,
        256,
        size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        dtype=np.uint8,
    )

    steering_angle = predict_steering(
        steering_model,
        dummy_image,
    )

    print("Inference pipeline tested successfully.")
    print("Loaded model:", model_path)
    print("Input image shape:", dummy_image.shape)
    print("Predicted steering angle:", steering_angle)