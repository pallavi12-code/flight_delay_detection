"""
Command-line inference: loads the trained model plus the exact encoders/
scaler/threshold fit during training, so a single flight's raw inputs are
transformed identically to how training data was and predicted (rather
than the placeholder zero-encoding used in early experiments).

Usage:
    python -m src.predict
"""
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config
from .data_preprocessing import encode_with_fallback
from .model import focal_loss


def load_artifacts():
    model = tf.keras.models.load_model(
        config.MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss()}
    )
    encoders = joblib.load(config.ENCODERS_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    threshold = joblib.load(config.THRESHOLD_PATH)
    return model, encoders, scaler, threshold


def predict_single(inputs: dict, model, encoders, scaler, threshold, T: int = config.SEQUENCE_LENGTH):
    """
    inputs: dict with all keys in config.NUM_COLS plus config.CAT_COLS
    (raw, unscaled/unencoded values).
    """
    num_row = pd.DataFrame([{c: inputs[c] for c in config.NUM_COLS}])
    num_row[config.NUM_COLS] = scaler.transform(num_row[config.NUM_COLS])
    num_features = num_row.values  # shape (1, n_features)

    cat_row = pd.DataFrame([{c: str(inputs[c]) for c in config.CAT_COLS}])
    for c in config.CAT_COLS:
        cat_row[c] = encode_with_fallback(encoders[c], cat_row[c])
    cat_features = cat_row.values  # shape (1, n_static)

    # No true history available for a single ad-hoc query, so repeat the
    # current reading T times to satisfy the LSTM's fixed sequence length.
    X_seq_input = np.repeat(num_features[:, np.newaxis, :], T, axis=1)

    prob = float(model.predict([X_seq_input, cat_features]).ravel()[0])
    label = "LIKELY DELAYED" if prob > threshold else "LIKELY ON TIME"
    return prob, label


def _prompt_for_inputs() -> dict:
    print("\n✈️  Enter flight details below:\n")
    values = {}
    prompts = {
        "departure_delay": "Departure delay (minutes): ",
        "delay_carrier": "Carrier delay (minutes): ",
        "delay_weather": "Weather delay (minutes): ",
        "delay_national_aviation_system": "National aviation system delay (minutes): ",
        "delay_security": "Security delay (minutes): ",
        "delay_late_aircarft_arrival": "Late aircraft delay (minutes): ",
        "HourlyDryBulbTemperature_x": "Origin temperature (°C): ",
        "HourlyVisibility_x": "Origin visibility (km): ",
        "HourlyWindSpeed_x": "Origin wind speed (km/h): ",
        "HourlyPrecipitation_x": "Origin precipitation (mm): ",
        "HourlyDryBulbTemperature_y": "Destination temperature (°C): ",
        "HourlyVisibility_y": "Destination visibility (km): ",
        "HourlyWindSpeed_y": "Destination wind speed (km/h): ",
        "HourlyPrecipitation_y": "Destination precipitation (mm): ",
    }
    for key, label in prompts.items():
        values[key] = float(input(label))
    values["weekday"] = int(input("Day of week (0=Mon, 6=Sun): "))
    values["carrier_code"] = input("Carrier code (e.g. AA, DL, UA): ")
    values["origin_airport"] = input("Origin airport (e.g. ATL, LAX): ")
    values["destination_airport"] = input("Destination airport (e.g. JFK, ORD): ")
    return values


if __name__ == "__main__":
    model, encoders, scaler, threshold = load_artifacts()
    user_inputs = _prompt_for_inputs()
    probability, prediction = predict_single(user_inputs, model, encoders, scaler, threshold)
    print("\n🧾 Prediction result:")
    print(f"Delay probability: {probability:.3f}")
    print(f"Predicted status : {prediction}")
