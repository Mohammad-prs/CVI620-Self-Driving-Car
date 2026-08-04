import base64
from io import BytesIO
from pathlib import Path

import cv2
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from tensorflow.keras.models import load_model

from src.data import preprocess_image


MODEL_PATH = Path("models/best_model.keras")
HOST = "0.0.0.0"
PORT = 4567

previous_steering = 0.0

# The model understeers on sharp left turns, so left predictions
# receive more gain than right predictions.
LEFT_STEERING_GAIN = 2.10
RIGHT_STEERING_GAIN = 1.25


SMOOTHING = 0.10


STRAIGHT_THROTTLE = 0.12
MEDIUM_TURN_THROTTLE = 0.055
SHARP_TURN_THROTTLE = 0.025

MEDIUM_TURN_THRESHOLD = 0.12
SHARP_TURN_THRESHOLD = 0.25

sio = socketio.Server()
app = socketio.Middleware(sio)


def send_control(steering_angle: float, throttle: float) -> None:
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
    )


@sio.on("connect")
def connect(sid, environ) -> None:
    global previous_steering

    previous_steering = 0.0
    print("Simulator connected:", sid)

    send_control(
        steering_angle=0.0,
        throttle=0.0,
    )


@sio.on("telemetry")
def telemetry(sid, data) -> None:
    global previous_steering

    if not data or "image" not in data:
        send_control(0.0, 0.0)
        return

    try:
        image = Image.open(
            BytesIO(
                base64.b64decode(data["image"])
            )
        ).convert("RGB")

        image_rgb = np.asarray(image)

        image_bgr = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        )

        processed_image = preprocess_image(
            image_bgr
        )

        image_batch = np.expand_dims(
            processed_image,
            axis=0,
        )

        model_prediction = float(
            model.predict(
                image_batch,
                verbose=0,
            )[0][0]
        )

        # Negative steering represents left turns in this dataset.
        if model_prediction < 0:
            corrected_steering = (
                model_prediction * LEFT_STEERING_GAIN
            )
        else:
            corrected_steering = (
                model_prediction * RIGHT_STEERING_GAIN
            )

        # React quickly without making the steering completely unstable.
        steering_angle = (
            SMOOTHING * previous_steering
            + (1.0 - SMOOTHING) * corrected_steering
        )

        steering_angle = float(
            np.clip(
                steering_angle,
                -1.0,
                1.0,
            )
        )

        previous_steering = steering_angle

        turn_strength = abs(steering_angle)

        if turn_strength >= SHARP_TURN_THRESHOLD:
            throttle = SHARP_TURN_THROTTLE
        elif turn_strength >= MEDIUM_TURN_THRESHOLD:
            throttle = MEDIUM_TURN_THROTTLE
        else:
            throttle = STRAIGHT_THROTTLE

        send_control(
            steering_angle,
            throttle,
        )

        speed = data.get("speed", "unknown")

        print(
            f"Prediction: {model_prediction: .4f} | "
            f"Steering: {steering_angle: .4f} | "
            f"Throttle: {throttle:.3f} | "
            f"Speed: {speed}"
        )

    except Exception as error:
        print("Telemetry error:", error)
        send_control(0.0, 0.0)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model was not found: {MODEL_PATH}"
    )


print("Loading model:", MODEL_PATH)

model = load_model(
    MODEL_PATH,
    safe_mode=False,
    compile=False,
)

print("Model loaded successfully.")
print(f"Starting simulator server on port {PORT}...")
print("Open the simulator and choose Autonomous Mode.")

eventlet.wsgi.server(
    eventlet.listen((HOST, PORT)),
    app,
)