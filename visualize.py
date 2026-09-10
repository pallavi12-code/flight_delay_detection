"""
Plotting helpers, kept separate from train.py so training can run headless
(e.g. in CI or a remote GPU box) without a display backend.
"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    from . import config
except ImportError:  # pragma: no cover
    import config


def plot_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png",
                          output_dir=config.DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Optimized Threshold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_training_history(history, filename: str = "training_history.png",
                          output_dir=config.DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path
