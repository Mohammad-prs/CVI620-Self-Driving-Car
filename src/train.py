from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config import BEST_MODEL_PATH, EPOCHS, LEARNING_RATE
from model import build_nvidia_model
from utils import create_output_directories, save_training_history


def create_callbacks():
    """
    Create callbacks used during model training.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
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
    )

    return model


def train_model(training_data, validation_data):
    """
    Train the steering-angle model.

    The real training and validation generators will be connected
    after the dataset pipeline is provided.
    """

    create_output_directories()

    model = prepare_model()
    callbacks = create_callbacks()

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_training_history(history.history)

    return model, history


if __name__ == "__main__":
    model = prepare_model()
    callbacks = create_callbacks()

    print("Training pipeline prepared successfully.")
    print("Model name:", model.name)
    print("Learning rate:", float(model.optimizer.learning_rate.numpy()))
    print("Loss function:", model.loss)
    print("Epochs:", EPOCHS)
    print("Callbacks:")

    for callback in callbacks:
        print("-", callback.__class__.__name__)