"""
Data loading, cleaning, encoding, scaling, and LSTM sequence construction.
"""
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config


def load_raw_data(data_dir: str = config.DATA_DIR) -> pd.DataFrame:
    """Concatenate every monthly CSV in the dataset directory."""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download the dataset first (see README) and unzip it here."
        )
    dfs = [pd.read_csv(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable columns and impute missing weather readings with the median."""
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    weather_cols = [c for c in df.columns if "Hourly" in c or "Precipitation" in c]
    for c in weather_cols:
        df[c] = df[c].fillna(df[c].median())

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Binary label: 1 if arrival delay exceeds the configured threshold."""
    df[config.TARGET] = (df["arrival_delay"] > config.DELAY_THRESHOLD_MINUTES).astype(int)
    return df


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select modeling columns and drop rows with any remaining missing values."""
    cols = config.NUM_COLS + config.CAT_COLS + [config.TARGET]
    return df[cols].dropna()


def encode_and_scale(df: pd.DataFrame, fit: bool = True, save: bool = True) -> pd.DataFrame:
    """
    Label-encode categorical columns and standard-scale numeric columns.
    When fit=True, new encoders/scaler are fit and (optionally) persisted to
    disk so predict.py can reuse the exact same transforms at inference time.
    """
    df = df.copy()
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if fit:
        encoders = {}
        for c in config.CAT_COLS:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le

        scaler = StandardScaler()
        df[config.NUM_COLS] = scaler.fit_transform(df[config.NUM_COLS])

        if save:
            joblib.dump(encoders, config.ENCODERS_PATH)
            joblib.dump(scaler, config.SCALER_PATH)
    else:
        encoders = joblib.load(config.ENCODERS_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        for c in config.CAT_COLS:
            df[c] = encode_with_fallback(encoders[c], df[c].astype(str))
        df[config.NUM_COLS] = scaler.transform(df[config.NUM_COLS])

    return df


def encode_with_fallback(label_encoder: LabelEncoder, values: pd.Series) -> np.ndarray:
    """
    LabelEncoder.transform raises on unseen categories (e.g. a new airport code).
    Map anything unseen to a reserved "unknown" bucket instead of crashing.
    """
    known = set(label_encoder.classes_)
    safe_values = values.apply(lambda v: v if v in known else label_encoder.classes_[0])
    return label_encoder.transform(safe_values)


def build_sequences(num_data: np.ndarray, cat_data: np.ndarray, y: np.ndarray, T: int = config.SEQUENCE_LENGTH):
    """
    Turn tabular rows into sliding-window sequences for the LSTM branch:
    each sample's sequence is the T rows preceding it, paired with that
    row's static (categorical) features and label.
    """
    X_seq, X_static, Y = [], [], []
    for i in range(T, len(num_data)):
        X_seq.append(num_data[i - T:i])
        X_static.append(cat_data[i])
        Y.append(y[i])

    return np.array(X_seq), np.array(X_static), np.array(Y)
