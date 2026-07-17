"""
Model definition: a two-branch network combining an LSTM over recent
sequential records with a dense branch over static (categorical) features,
trained with focal loss to handle class imbalance (delays are the minority class).
"""
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras import backend as K

from . import config


def focal_loss(gamma: float = config.FOCAL_LOSS_GAMMA, alpha: float = config.FOCAL_LOSS_ALPHA):
    """
    Focal loss down-weights easy, already-well-classified examples so the
    model keeps learning from the harder, minority-class (delayed) samples.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = (
            alpha * y_true * tf.pow(1 - y_pred, gamma)
            + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        )
        return tf.reduce_mean(weight * cross_entropy)

    return focal_loss_fixed


def build_model(T: int, n_features: int, n_static: int) -> Model:
    """Build and compile the LSTM + static-feature model."""
    seq_input = Input(shape=(T, n_features), name="sequence_input")
    x = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(seq_input)
    x = layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)(x)
    x = layers.BatchNormalization()(x)

    static_input = Input(shape=(n_static,), name="static_input")
    s = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(static_input)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(0.3)(s)

    combined = layers.concatenate([x, s])
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.4)(z)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = Model(inputs=[seq_input, static_input], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=focal_loss(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
