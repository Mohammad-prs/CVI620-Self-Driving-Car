import random
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import (
    BATCH_SIZE,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    TEST_IMAGES_DIR,
    TEST_LOG_PATH,
    TRAIN_IMAGES_DIR,
    TRAIN_LOG_PATH,
)


def load_driving_log(csv_path: Path, images_dir: Path) -> pd.DataFrame:
    """
    Load a driving log and replace computer-specific image paths
    with valid local paths from this project.
    """

    data = pd.read_csv(csv_path, sep=";")

    required_columns = {"Imgs", "Steering"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"CSV must contain {required_columns}, "
            f"but found {data.columns.tolist()}."
        )

    data = data[["Imgs", "Steering"]].copy()

    # The CSV contains absolute Windows paths from Joseph's computer.
    # Keep only the filename and rebuild the path using this project.
    data["image_path"] = data["Imgs"].apply(
        lambda path: images_dir / Path(str(path).replace("\\", "/")).name
    )

    data["Steering"] = pd.to_numeric(
        data["Steering"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["Steering"],
    ).reset_index(drop=True)

    missing_images = data[
        ~data["image_path"].apply(Path.exists)
    ]

    if not missing_images.empty:
        example = missing_images["image_path"].iloc[0]

        raise FileNotFoundError(
            f"{len(missing_images)} images referenced by the CSV "
            f"could not be found. Example: {example}"
        )

    return data


def balance_training_data(
    data: pd.DataFrame,
    near_zero_threshold: float = 0.05,
    max_near_zero_samples: int = 1500,
) -> pd.DataFrame:
    """
    Reduce the excessive number of nearly straight-driving samples.

    All turning samples are preserved. Only near-zero steering samples
    are randomly reduced.
    """

    if data.empty:
        raise ValueError("The training dataset is empty.")

    near_zero_data = data[
        data["Steering"].abs() < near_zero_threshold
    ]

    turning_data = data[
        data["Steering"].abs() >= near_zero_threshold
    ]

    if len(near_zero_data) > max_near_zero_samples:
        near_zero_data = near_zero_data.sample(
            n=max_near_zero_samples,
            random_state=RANDOM_SEED,
        )

    balanced_data = pd.concat(
        [near_zero_data, turning_data],
        ignore_index=True,
    )

    balanced_data = balanced_data.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return balanced_data


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare one BGR simulator image for the Nvidia CNN.

    Steps:
    1. Crop the road region.
    2. Convert BGR to YUV.
    3. Apply Gaussian blur.
    4. Resize to 200 x 66.

    Normalization is handled by model.py.
    """

    if image is None:
        raise ValueError("The image could not be loaded.")

    height, width = image.shape[:2]

    cropped = image[
        int(height * 0.35):int(height * 0.97),
        int(width * 0.05):int(width * 0.95),
    ]

    if cropped.size == 0:
        raise ValueError("Image crop produced an empty image.")

    yuv = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2YUV,
    )

    blurred = cv2.GaussianBlur(
        yuv,
        (3, 3),
        0,
    )

    resized = cv2.resize(
        blurred,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype(np.float32)


def augment_image(
    image: np.ndarray,
    steering_angle: float,
) -> Tuple[np.ndarray, float]:
    """
    Apply random augmentation to a training image.
    """

    # Horizontal flip.
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle *= -1.0

    # Random brightness.
    if random.random() < 0.5:
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        ).astype(np.float32)

        brightness = random.uniform(0.7, 1.3)

        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] * brightness,
            0,
            255,
        )

        image = cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2BGR,
        )

    return image, steering_angle


def batch_generator(
    data: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Continuously generate image and steering-angle batches.
    """

    if data.empty:
        raise ValueError("The dataset is empty.")

    while True:
        indices = np.arange(len(data))

        if training:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]

            images = []
            steering_angles = []

            for index in batch_indices:
                row = data.iloc[index]

                image = cv2.imread(
                    str(row["image_path"])
                )

                if image is None:
                    raise FileNotFoundError(
                        f"Could not read image: {row['image_path']}"
                    )

                steering_angle = float(
                    row["Steering"]
                )

                if training:
                    image, steering_angle = augment_image(
                        image,
                        steering_angle,
                    )

                image = preprocess_image(image)

                images.append(image)
                steering_angles.append(steering_angle)

            yield (
                np.asarray(images, dtype=np.float32),
                np.asarray(steering_angles, dtype=np.float32),
            )


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the balanced training dataset and unchanged test dataset.
    """

    train_data = load_driving_log(
        TRAIN_LOG_PATH,
        TRAIN_IMAGES_DIR,
    )

    train_data = balance_training_data(
        train_data,
    )

    test_data = load_driving_log(
        TEST_LOG_PATH,
        TEST_IMAGES_DIR,
    )

    return train_data, test_data


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_data, test_data = load_datasets()

    steering = train_data["Steering"]

    print("Balanced training rows:", len(train_data))
    print("Test rows:", len(test_data))
    print("Near-zero training rows:", (steering.abs() < 0.05).sum())
    print("Turning training rows:", (steering.abs() >= 0.05).sum())
    print("Negative steering rows:", (steering < 0).sum())
    print("Positive steering rows:", (steering > 0).sum())
    print("First training image:", train_data["image_path"].iloc[0])

    generator = batch_generator(
        train_data,
        batch_size=4,
        training=True,
    )

    image_batch, steering_batch = next(generator)

    print("Image batch shape:", image_batch.shape)
    print("Steering batch shape:", steering_batch.shape)
    print("Image dtype:", image_batch.dtype)
    print("Image value range:", image_batch.min(), image_batch.max())