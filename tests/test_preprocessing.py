import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

import config
from data_preprocessing import (
    Preprocessor,
    build_sequences,
    build_model_frame,
    encode_with_fallback,
)
from evaluation import evaluate_predictions


def sample_frame(rows=8):
    data = {column: np.arange(rows, dtype=float) for column in config.NUM_COLS}
    data.update({
        "carrier_code": ["AA", "DL"] * (rows // 2),
        "origin_airport": ["ATL", "LAX"] * (rows // 2),
        "destination_airport": ["JFK", "ORD"] * (rows // 2),
        config.TARGET: [0, 1] * (rows // 2),
    })
    return pd.DataFrame(data)


def test_unseen_categories_use_known_bucket():
    encoder = LabelEncoder().fit(["AA", "DL"])
    result = encode_with_fallback(encoder, pd.Series(["AA", "NEW"]))
    assert result.tolist() == [0, 0]


def test_preprocessor_round_trip(tmp_path):
    frame = sample_frame()
    preprocessor = Preprocessor.fit(frame)
    preprocessor.save(tmp_path)
    restored = Preprocessor.load(tmp_path)
    transformed = restored.transform(frame)
    assert transformed[config.NUM_COLS].shape == (len(frame), len(config.NUM_COLS))
    assert transformed[config.CAT_COLS].dtypes.tolist() == [np.dtype("int64")] * 3


def test_model_frame_reports_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        build_model_frame(pd.DataFrame())


def test_build_sequences_aligns_current_label():
    numeric = np.arange(20).reshape(10, 2)
    categorical = np.arange(10).reshape(10, 1)
    labels = np.arange(10)
    sequences, static, target = build_sequences(numeric, categorical, labels, T=3)
    assert sequences.shape == (7, 3, 2)
    assert static[0].tolist() == [3]
    assert target.tolist() == list(range(3, 10))


def test_build_sequences_rejects_short_input():
    with pytest.raises(ValueError):
        build_sequences(np.ones((2, 1)), np.ones((2, 1)), np.ones(2), T=2)


def test_evaluate_predictions_validates_lengths():
    metrics = evaluate_predictions([0, 1], [0.1, 0.9], threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    with pytest.raises(ValueError):
        evaluate_predictions([0], [0.1, 0.2], threshold=0.5)
