"""Configuration for the flight-delay training and inference pipeline."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "flight_weather_data"
DEFAULT_MODEL_DIR = ROOT_DIR / "models"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"

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

SEQUENCE_LENGTH = 5
SAMPLE_SIZE = 80_000
RANDOM_STATE = 42
TRAIN_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 5
LR_PLATEAU_PATIENCE = 3
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25


@dataclass(frozen=True)
class Settings:
    """Runtime settings, with paths kept together for artifact consistency."""

    data_dir: Path = DEFAULT_DATA_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    random_state: int = RANDOM_STATE
    sample_size: int = SAMPLE_SIZE
    epochs: int = TRAIN_EPOCHS
    batch_size: int = BATCH_SIZE
    sequence_length: int = SEQUENCE_LENGTH

    @property
    def model_path(self) -> Path:
        return self.model_dir / "flight_delay_model.keras"

    @property
    def encoders_path(self) -> Path:
        return self.model_dir / "label_encoders.joblib"

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / "scaler.joblib"

    @property
    def threshold_path(self) -> Path:
        return self.model_dir / "best_threshold.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"
