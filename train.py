"""Train the LSTM model with a chronological, leakage-safe data split.

Run with ``python train.py --help`` or ``python -m train --help``.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils import class_weight

try:
    from . import config
    from .data_preprocessing import (
        Preprocessor, build_model_frame, build_sequences, clean_data,
        create_target, load_raw_data,
    )
    from .model import build_model
    from .evaluation import evaluate_predictions
    from .visualize import plot_confusion_matrix, plot_training_history
except ImportError:  # pragma: no cover
    import config
    from data_preprocessing import (
        Preprocessor, build_model_frame, build_sequences, clean_data,
        create_target, load_raw_data,
    )
    from model import build_model
    from evaluation import evaluate_predictions
    from visualize import plot_confusion_matrix, plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=config.DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=config.SAMPLE_SIZE)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def split_frame(df, sample_size: int):
    """Keep source order: random row splits make LSTM history unrealistic."""
    if sample_size < 3:
        raise ValueError("sample_size must be at least 3.")
    frame = df.iloc[:min(sample_size, len(df))].reset_index(drop=True)
    if len(frame) < 3:
        raise ValueError("At least three rows are required for train/validation/test splits.")
    train_end = int(len(frame) * 0.6)
    val_end = int(len(frame) * 0.8)
    if train_end == 0 or val_end <= train_end or val_end == len(frame):
        raise ValueError("Dataset is too small for chronological splits.")
    return frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:]


def make_inputs(frame, preprocessor, sequence_length):
    transformed = preprocessor.transform(frame)
    return build_sequences(
        transformed[config.NUM_COLS].to_numpy(),
        transformed[config.CAT_COLS].to_numpy(),
        transformed[config.TARGET].to_numpy(),
        T=sequence_length,
    )


def choose_threshold(y_true, probabilities) -> float:
    if len(np.unique(y_true)) < 2:
        raise ValueError("Validation data must contain both target classes.")
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return float(thresholds[np.argmax(tpr - fpr)])


def main(args=None):
    args = args or parse_args()
    set_seed(args.seed)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")

    frame = build_model_frame(create_target(clean_data(load_raw_data(args.data_dir))))
    train_frame, val_frame, test_frame = split_frame(frame, args.sample_size)
    preprocessor = Preprocessor.fit(train_frame)
    preprocessor.save(args.model_dir)

    train_inputs = make_inputs(train_frame, preprocessor, args.sequence_length)
    val_inputs = make_inputs(val_frame, preprocessor, args.sequence_length)
    test_inputs = make_inputs(test_frame, preprocessor, args.sequence_length)
    X_seq, X_static, y = train_inputs
    X_val_seq, X_val_static, y_val = val_inputs

    weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )
    class_weights = dict(zip(np.unique(y), weights))

    model = build_model(X_seq.shape[1], X_seq.shape[2], X_static.shape[1])
    import tensorflow as tf
    history = model.fit(
        [X_seq, X_static], y,
        validation_data=([X_val_seq, X_val_static], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=config.LR_PLATEAU_PATIENCE,
                min_lr=1e-6,
            ),
        ],
        verbose=1,
    )

    val_prob = model.predict([X_val_seq, X_val_static], verbose=0).ravel()
    threshold = choose_threshold(y_val, val_prob)
    X_test_seq, X_test_static, y_test = test_inputs
    test_prob = model.predict([X_test_seq, X_test_static], verbose=0).ravel()
    test_pred = (test_prob >= threshold).astype(int)
    metrics = evaluate_predictions(y_test, test_prob, threshold)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir / "flight_delay_model.keras")
    joblib.dump(threshold, args.model_dir / "best_threshold.joblib")
    (args.model_dir / "metadata.json").write_text(json.dumps({
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "num_features": len(config.NUM_COLS),
        "static_features": len(config.CAT_COLS),
        "data_split": "chronological 60/20/20",
    }, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(y_test, test_pred, output_dir=args.output_dir)
    plot_training_history(history, output_dir=args.output_dir)
    print(json.dumps({"test_roc_auc": metrics["roc_auc"], "threshold": threshold}, indent=2))
    return history


if __name__ == "__main__":
    main()
