"""
The four architecture families compared in this study.

All four consume the identical (SEQ_LEN=30, F=14) window and emit a single
linear RUL scalar, are trained with the identical optimizer/schedule, and are
quantized with the identical calibration procedure. The architecture is the
only independent variable.

    cnn          -- the incumbent. Byte-for-byte the same topology as
                    Code/train_model.py, so the enhanced study reproduces the
                    original result rather than replacing it.
    lstm         -- recurrent. The family with the strongest a-priori reason
                    to suffer under static INT8: activation quantization error
                    is re-injected into the hidden state at every one of the
                    30 timesteps rather than passing through a fixed-depth
                    feed-forward stack once.
    transformer  -- attention. Softmax and LayerNormalization are the two ops
                    usually cited as hostile to per-tensor INT8.
    mlp          -- control. Flattens the window and discards temporal
                    structure entirely; a pure feed-forward stack is the
                    easiest possible case for post-training quantization and
                    acts as the floor for "how little can quantization cost".

NOTE ON `unroll=True` FOR THE LSTM
    This is not a modelling preference, it is a hard toolchain constraint that
    the study measures. With the default `unroll=False`, TensorFlow 2.20's
    TFLite converter emits tf.TensorListReserve ops and fails outright:

        error: 'tf.TensorListReserve' op requires element_shape to be static
        during TF Lite transformation pass

    Falling back to SELECT_TF_OPS produces a flatbuffer that the standard
    interpreter refuses to run without the Flex delegate, which does not exist
    on microcontroller targets. Unrolling the 30 timesteps is what makes a
    fully-INT8 recurrent model convertible at all -- at a real and reportable
    cost in flatbuffer size, since the cell graph is duplicated 30 times.
    run_study.py records both facts.
"""

import tensorflow as tf

from common import SEQ_LEN, FEATURES, LEARNING_RATE

N_FEATURES = len(FEATURES)


def _compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def build_cnn(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """Identical topology to Code/train_model.py."""
    i = tf.keras.Input(shape=(seq_len, n_features), name="window")
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation="relu")(i)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    o = tf.keras.layers.Dense(1, name="rul")(x)
    return _compile(tf.keras.Model(i, o, name="cnn"))


def build_lstm(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """Stacked LSTM. `unroll=True` is required for INT8 conversion (see above)."""
    i = tf.keras.Input(shape=(seq_len, n_features), name="window")
    x = tf.keras.layers.LSTM(64, return_sequences=True, unroll=True)(i)
    x = tf.keras.layers.LSTM(32, unroll=True)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    o = tf.keras.layers.Dense(1, name="rul")(x)
    return _compile(tf.keras.Model(i, o, name="lstm"))


def build_transformer(seq_len=SEQ_LEN, n_features=N_FEATURES,
                      d_model=32, num_heads=2, ff_dim=64):
    """Single pre-projected Transformer encoder block + global average pooling."""
    i = tf.keras.Input(shape=(seq_len, n_features), name="window")
    x = tf.keras.layers.Dense(d_model)(i)

    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)

    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dense(d_model)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    o = tf.keras.layers.Dense(1, name="rul")(x)
    return _compile(tf.keras.Model(i, o, name="transformer"))


def build_mlp(seq_len=SEQ_LEN, n_features=N_FEATURES):
    """Temporal-structure-free control."""
    i = tf.keras.Input(shape=(seq_len, n_features), name="window")
    x = tf.keras.layers.Flatten()(i)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    o = tf.keras.layers.Dense(1, name="rul")(x)
    return _compile(tf.keras.Model(i, o, name="mlp"))


BUILDERS = {
    "cnn": build_cnn,
    "lstm": build_lstm,
    "transformer": build_transformer,
    "mlp": build_mlp,
}

# Display order and labels used by make_figures.py
ORDER = ["cnn", "lstm", "transformer", "mlp"]
LABELS = {
    "cnn": "1D-CNN\n(incumbent)",
    "lstm": "LSTM\n(recurrent)",
    "transformer": "Transformer\n(attention)",
    "mlp": "MLP\n(no temporal)",
}
