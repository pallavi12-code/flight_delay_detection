"""
Command-line inference: loads the trained model plus the exact encoders/
scaler/threshold fit during training, so a single flight's raw inputs are
transformed identically to how training data was and predicted (rather
than the placeholder zero-encoding used in early experiments).

Usage:
    python -m src.predict
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from . import config
    from .data_preprocessing import Preprocessor
    from .model import focal_loss
except ImportError:  # pragma: no cover
    import config
    from data_preprocessing import Preprocessor
    from model import focal_loss


def load_artifacts(model_dir: Path = config.DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    required = [
        model_dir / "flight_delay_model.keras",
        model_dir / "best_threshold.joblib",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")
    model = tf.keras.models.load_model(
        model_dir / "flight_delay_model.keras",
        custom_objects={"focal_loss_fixed": focal_loss()},
    )
    preprocessor = Preprocessor.load(model_dir)
    import joblib
    threshold = float(joblib.load(model_dir / "best_threshold.joblib"))
    return model, preprocessor, threshold


def predict_single(inputs: dict, model, preprocessor, threshold,
                   T: int = config.SEQUENCE_LENGTH):
    """
    inputs: dict with all keys in config.NUM_COLS plus config.CAT_COLS
    (raw, unscaled/unencoded values).
    """
    missing = sorted(set(config.NUM_COLS + config.CAT_COLS) - set(inputs))
    if missing:
        raise ValueError(f"Missing prediction inputs: {missing}")
    num_row = pd.DataFrame([{c: inputs[c] for c in config.NUM_COLS + config.CAT_COLS}])
    transformed = preprocessor.transform(num_row)
    num_features = transformed[config.NUM_COLS].to_numpy()
    cat_features = transformed[config.CAT_COLS].to_numpy()

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
    parser = argparse.ArgumentParser(description="Predict one flight delay probability.")
    parser.add_argument("--model-dir", type=Path, default=config.DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    model, preprocessor, threshold = load_artifacts(args.model_dir)
    user_inputs = _prompt_for_inputs()
    probability, prediction = predict_single(user_inputs, model, preprocessor, threshold)
    print("\n🧾 Prediction result:")
    print(f"Delay probability: {probability:.3f}")
    print(f"Predicted status : {prediction}")
