"""Data loading, leakage-safe preprocessing, and LSTM sequence construction."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from . import config
except ImportError:  # pragma: no cover - supports ``python train.py``
    import config


def load_raw_data(data_dir: Path = config.DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load CSV files in deterministic order."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove excluded columns without learning statistics from future rows."""
    return df.drop(columns=[c for c in config.DROP_COLS if c in df], errors="ignore").copy()


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the delay label without mutating the caller's frame."""
    if "arrival_delay" not in df:
        raise ValueError("Input data must contain 'arrival_delay' to create the target.")
    result = df.copy()
    result[config.TARGET] = (result["arrival_delay"] > config.DELAY_THRESHOLD_MINUTES).astype(int)
    return result


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select required columns and remove rows still containing missing values."""
    columns = config.NUM_COLS + config.CAT_COLS + [config.TARGET]
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    return df[columns].dropna(subset=[config.TARGET] + config.CAT_COLS).reset_index(drop=True)


class Preprocessor:
    """Persisted train-only categorical encoders and numeric scaler."""

    def __init__(self, encoders: Dict[str, LabelEncoder], scaler: StandardScaler,
                 numeric_medians: pd.Series):
        self.encoders = encoders
        self.scaler = scaler
        self.numeric_medians = numeric_medians

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Preprocessor":
        encoders = {}
        for column in config.CAT_COLS:
            encoder = LabelEncoder()
            encoder.fit(df[column].astype(str))
            encoders[column] = encoder
        numeric_medians = df[config.NUM_COLS].median()
        scaler = StandardScaler().fit(df[config.NUM_COLS].fillna(numeric_medians))
        return cls(encoders, scaler, numeric_medians)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in config.CAT_COLS:
            result[column] = encode_with_fallback(self.encoders[column], result[column].astype(str))
        result[config.NUM_COLS] = self.scaler.transform(
            result[config.NUM_COLS].fillna(self.numeric_medians)
        )
        return result

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.encoders, model_dir / "label_encoders.joblib")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(self.numeric_medians, model_dir / "numeric_medians.joblib")

    @classmethod
    def load(cls, model_dir: Path) -> "Preprocessor":
        try:
            encoders = joblib.load(model_dir / "label_encoders.joblib")
            scaler = joblib.load(model_dir / "scaler.joblib")
            numeric_medians = joblib.load(model_dir / "numeric_medians.joblib")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Preprocessing artifact is missing in {model_dir}.") from exc
        return cls(encoders, scaler, numeric_medians)


def encode_with_fallback(label_encoder: LabelEncoder, values: pd.Series) -> np.ndarray:
    """Encode unseen categories as the first known category instead of failing."""
    known = set(label_encoder.classes_)
    safe_values = values.map(lambda value: value if value in known else label_encoder.classes_[0])
    return label_encoder.transform(safe_values)


def encode_and_scale(df: pd.DataFrame, fit: bool = True, save: bool = True,
                     model_dir: Path = config.DEFAULT_MODEL_DIR) -> pd.DataFrame:
    """Compatibility wrapper around :class:`Preprocessor`."""
    preprocessor = Preprocessor.fit(df) if fit else Preprocessor.load(model_dir)
    if fit and save:
        preprocessor.save(model_dir)
    return preprocessor.transform(df)


def build_sequences(num_data: np.ndarray, cat_data: np.ndarray, y: np.ndarray,
                    T: int = config.SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build preceding-row windows and align each window with its current label."""
    if T < 1 or len(num_data) <= T:
        raise ValueError("At least T + 1 rows are required to build sequences.")
    if not (len(num_data) == len(cat_data) == len(y)):
        raise ValueError("Numeric features, categorical features, and labels must have equal length.")
    return (
        np.asarray([num_data[i - T:i] for i in range(T, len(num_data))]),
        np.asarray([cat_data[i] for i in range(T, len(num_data))]),
        np.asarray([y[i] for i in range(T, len(num_data))]),
    )
