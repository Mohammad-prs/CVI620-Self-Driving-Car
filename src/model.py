import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
)


def build_nvidia_model() -> Sequential:
    """
    Build the Nvidia CNN architecture for steering-angle prediction.

    Input:
        (66, 200, 3)

    Output:
        Single steering angle
    """

    model = Sequential(
        [
            # Normalize image from [0,255] -> [-1,1]
            Lambda(
                lambda image: image / 127.5 - 1.0,
                input_shape=(66, 200, 3),
                name="normalization",
            ),

            Conv2D(
                filters=24,
                kernel_size=(5, 5),
                strides=(2, 2),
                activation="elu",
                name="conv_1",
            ),

            Conv2D(
                filters=36,
                kernel_size=(5, 5),
                strides=(2, 2),
                activation="elu",
                name="conv_2",
            ),

            Conv2D(
                filters=48,
                kernel_size=(5, 5),
                strides=(2, 2),
                activation="elu",
                name="conv_3",
            ),

            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="elu",
                name="conv_4",
            ),

            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="elu",
                name="conv_5",
            ),

            Flatten(name="flatten"),

            Dense(100, activation="elu", name="dense_1"),
            Dropout(0.5, name="dropout_1"),

            Dense(50, activation="elu", name="dense_2"),
            Dropout(0.5, name="dropout_2"),

            Dense(10, activation="elu", name="dense_3"),

            # Regression output
            Dense(1, name="steering_output"),
        ],
        name="nvidia_steering_model",
    )

    return model


if __name__ == "__main__":
    steering_model = build_nvidia_model()

    steering_model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    steering_model.summary()

    print("\nModel compiled successfully.")
    print("Loss function:", steering_model.loss)
    print("Optimizer:", steering_model.optimizer.__class__.__name__)

    dummy_images = np.random.randint(
        0,
        256,
        size=(2, 66, 200, 3),
        dtype=np.uint8,
    )

    predictions = steering_model.predict(
        dummy_images,
        verbose=0,
    )

    print("\nDummy input shape:", dummy_images.shape)
    print("Prediction shape:", predictions.shape)

    model_path = "models/untrained_nvidia_model.keras"
    steering_model.save(model_path)

    print("\nModel saved successfully.")
    print("Saved path:", model_path)