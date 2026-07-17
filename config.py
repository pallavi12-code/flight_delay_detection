"""
Central configuration: paths, feature lists, and hyperparameters.
Keeping these in one place means every script (train/evaluate/predict)
stays in sync instead of drifting apart.
"""
import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "flight_weather_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "flight_delay_model.keras")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "best_threshold.joblib")

# ---------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------
NUM_COLS = [
    "departure_delay",
    "delay_carrier",
    "delay_weather",
    "delay_national_aviation_system",
    "delay_security",
    "delay_late_aircarft_arrival",
    "HourlyDryBulbTemperature_x",
    "HourlyVisibility_x",
    "HourlyWindSpeed_x",
    "HourlyPrecipitation_x",
    "HourlyDryBulbTemperature_y",
    "HourlyVisibility_y",
    "HourlyWindSpeed_y",
    "HourlyPrecipitation_y",
    "weekday",
]

CAT_COLS = ["carrier_code", "origin_airport", "destination_airport"]
TARGET = "DELAYED"
DELAY_THRESHOLD_MINUTES = 15
DROP_COLS = ["actual_arrival_dt", "actual_departure_dt", "tail_number"]

# ---------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------
SEQUENCE_LENGTH = 5          # T: how many past records form one LSTM sequence
SAMPLE_SIZE = 80_000         # rows sampled before sequence-building (memory/compute control)
RANDOM_STATE = 42

FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

TRAIN_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 5
LR_PLATEAU_PATIENCE = 3
