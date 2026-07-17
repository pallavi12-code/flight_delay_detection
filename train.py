"""
End-to-end training entry point.

Usage:
    python -m src.train
"""
import os

import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from . import config
from .data_preprocessing import (
    build_model_frame,
    build_sequences,
    clean_data,
    create_target,
    encode_and_scale,
    load_raw_data,
)
from .model import build_model
from .visualize import plot_confusion_matrix, plot_training_history


def main():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Total records: {len(df)}")

    df = clean_data(df)
    df = create_target(df)
    df_model = build_model_frame(df)
    print(f"Modeling frame shape: {df_model.shape}")

    df_model = encode_and_scale(df_model, fit=True, save=True)

    df_sample = df_model.sample(n=min(config.SAMPLE_SIZE, len(df_model)), random_state=config.RANDOM_STATE)
    print(f"Sampled shape: {df_sample.shape}")

    X = df_sample[config.NUM_COLS + config.CAT_COLS]
    y = df_sample[config.TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=config.RANDOM_STATE
    )

    num_data = X_train[config.NUM_COLS].values
    cat_data = X_train[config.CAT_COLS].values
    X_seq, X_static, Y = build_sequences(num_data, cat_data, y_train, T=config.SEQUENCE_LENGTH)
    print("LSTM input shape:", X_seq.shape)
    print("Static input shape:", X_static.shape)

    X_seq_train, X_seq_val, X_static_train, X_static_val, Y_train, Y_val = train_test_split(
        X_seq, X_static, Y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=Y
    )

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(Y_train), y=Y_train
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    model = build_model(T=X_seq.shape[1], n_features=X_seq.shape[2], n_static=X_static.shape[1])
    model.summary()

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=config.LR_PLATEAU_PATIENCE, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        [X_seq_train, X_static_train],
        Y_train,
        validation_data=([X_seq_val, X_static_val], Y_val),
        epochs=config.TRAIN_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
    )

    # Pick an operating threshold from the validation set (Youden's J), then
    # report final metrics on the full held-out sequence set.
    y_val_prob = model.predict([X_seq_val, X_static_val]).ravel()
    fpr, tpr, thresholds = roc_curve(Y_val, y_val_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print("\nBest threshold found:", best_threshold)

    y_pred_prob_all = model.predict([X_seq, X_static]).ravel()
    y_pred_opt = (y_pred_prob_all > best_threshold).astype(int)

    print("\nClassification report (optimized threshold):\n", classification_report(Y, y_pred_opt))
    print("ROC AUC:", roc_auc_score(Y, y_pred_prob_all))

    cm_path = plot_confusion_matrix(Y, y_pred_opt)
    hist_path = plot_training_history(history)
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved training history plot to {hist_path}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model.save(config.MODEL_PATH)
    joblib.dump(float(best_threshold), config.THRESHOLD_PATH)
    print(f"\nSaved model to {config.MODEL_PATH}")
    print(f"Saved threshold to {config.THRESHOLD_PATH}")

    return history


if __name__ == "__main__":
    main()
