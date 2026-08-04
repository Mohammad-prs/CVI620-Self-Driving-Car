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
    Load a driving log and rebuild image paths locally.
    """

    data = pd.read_csv(csv_path, sep=";")

    required_columns = {"Imgs", "Steering"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"CSV must contain {required_columns}, "
            f"but found {data.columns.tolist()}."
        )

    data = data[["Imgs", "Steering"]].copy()

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
            f"{len(missing_images)} missing images. Example: {example}"
        )

    return data


def balance_training_data(
    data: pd.DataFrame,
    near_zero_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Oversample turning examples instead of deleting straight images.
    """

    if data.empty:
        raise ValueError("Training dataset is empty.")

    straight = data[
        data["Steering"].abs() < near_zero_threshold
    ]

    turning = data[
        data["Steering"].abs() >= near_zero_threshold
    ]

    medium_turns = turning[
        turning["Steering"].abs() < 0.40
    ]

    strong_turns = turning[
        turning["Steering"].abs() >= 0.40
    ]

    balanced = pd.concat(
        [
            straight,

            turning,

            # duplicate every turning image
            turning,

            # duplicate medium turns again
            medium_turns,

            # duplicate sharp turns 3 extra times
            strong_turns,
            strong_turns,
            strong_turns,
        ],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    return balanced


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare one image for the Nvidia model.
    """

    if image is None:
        raise ValueError("Image could not be loaded.")

    height, width = image.shape[:2]

    cropped = image[
        int(height * 0.35):int(height * 0.97),
        int(width * 0.05):int(width * 0.95),
    ]

    if cropped.size == 0:
        raise ValueError("Crop produced an empty image.")

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

    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle *= -1.0

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

    if random.random() < 0.5:
        shift = random.randint(-50, 50)

        matrix = np.float32(
            [
                [1, 0, shift],
                [0, 1, 0],
            ]
        )

        image = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
        )

        steering_angle += shift * 0.002

    if random.random() < 0.3:
        zoom = random.uniform(1.0, 1.2)

        h, w = image.shape[:2]

        nh = int(h / zoom)
        nw = int(w / zoom)

        y = random.randint(0, h - nh)
        x = random.randint(0, w - nw)

        image = image[y:y + nh, x:x + nw]

        image = cv2.resize(
            image,
            (w, h),
        )

    return image, steering_angle


def batch_generator(
    data: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

    while True:

        indices = np.arange(len(data))

        if training:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):

            batch_indices = indices[start:start + batch_size]

            images = []
            steering = []

            for idx in batch_indices:

                row = data.iloc[idx]

                image = cv2.imread(
                    str(row["image_path"])
                )

                angle = float(
                    row["Steering"]
                )

                if training:
                    image, angle = augment_image(
                        image,
                        angle,
                    )

                image = preprocess_image(image)

                images.append(image)
                steering.append(angle)

            yield (
                np.asarray(images, dtype=np.float32),
                np.asarray(steering, dtype=np.float32),
            )


def load_datasets():

    train1 = load_driving_log(
        TRAIN_LOG_PATH,
        TRAIN_IMAGES_DIR,
    )

    train2 = load_driving_log(
        TEST_LOG_PATH,
        TEST_IMAGES_DIR,
    )

    combined = pd.concat(
        [train1, train2],
        ignore_index=True,
    )

    combined = combined.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    split = int(len(combined) * 0.9)

    train_data = combined.iloc[:split].reset_index(drop=True)
    validation_data = combined.iloc[split:].reset_index(drop=True)

    train_data = balance_training_data(
        train_data
    )

    return train_data, validation_data


if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_data, validation_data = load_datasets()

    steering = train_data["Steering"]

    print("Training rows:", len(train_data))
    print("Validation rows:", len(validation_data))
    print("Near zero:", (steering.abs() < 0.05).sum())
    print("Turning:", (steering.abs() >= 0.05).sum())
    print("Negative:", (steering < 0).sum())
    print("Positive:", (steering > 0).sum())