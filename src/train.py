import math
import random

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FINAL_MODEL_PATH,
    LEARNING_RATE,
    RANDOM_SEED,
)
from src.data import batch_generator, load_datasets
from src.model import build_nvidia_model
from src.utils import create_output_directories, save_training_history


def create_callbacks():
    """
    Create callbacks used during model training.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    return [early_stopping, model_checkpoint]


def prepare_model():
    """
    Build and compile the Nvidia steering-angle model.
    """

    model = build_nvidia_model()

    optimizer = Adam(
        learning_rate=LEARNING_RATE,
    )

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    return model


def train_model():
    """
    Load the datasets, train the model and save the results.
    """

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    create_output_directories()

    train_data, validation_data = load_datasets()

    training_generator = batch_generator(
        train_data,
        batch_size=BATCH_SIZE,
        training=True,
    )

    validation_generator = batch_generator(
        validation_data,
        batch_size=BATCH_SIZE,
        training=False,
    )

    steps_per_epoch = math.ceil(
        len(train_data) / BATCH_SIZE
    )

    validation_steps = math.ceil(
        len(validation_data) / BATCH_SIZE
    )

    model = prepare_model()
    callbacks = create_callbacks()

    print("Starting model training...")
    print("Training samples:", len(train_data))
    print("Validation samples:", len(validation_data))
    print("Batch size:", BATCH_SIZE)
    print("Steps per epoch:", steps_per_epoch)
    print("Validation steps:", validation_steps)

    history = model.fit(
        training_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_MODEL_PATH)

    save_training_history(
        history.history
    )

    print("\nTraining completed.")
    print("Best model:", BEST_MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)

    return model, history


if __name__ == "__main__":
    train_model()