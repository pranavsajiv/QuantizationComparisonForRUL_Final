"""
Shared data pipeline, metrics and quantization utilities for the AeroGuard
architecture x quantization study.

Every preprocessing choice here is copied deliberately from the original
Code/train_model.py so that the ONLY thing differing between the four
architectures compared in run_study.py is the network itself:

    - RUL target capped at 130 cycles (piecewise-linear degradation)
    - the same 14 retained sensors after variance-based feature masking
    - Min-Max scaling with bounds fit on the training split ONLY
    - sliding window of L = 30 cycles
    - GroupShuffleSplit(test_size=0.1) grouped by engine unit
    - Adam(1e-3), batch 64, up to 50 epochs, EarlyStopping(patience=5)

Nothing in this module invents a number. Everything reported by run_study.py
is computed from a model that was actually trained and an interpreter that
was actually invoked.
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------------------------------------------
# Constants -- identical to Code/train_model.py
# --------------------------------------------------------------------------
COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]

FEATURES = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
            "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

SEQ_LEN = 30
RUL_CAP = 130

# Operational alert thresholds (Eq. 17 in the report; also the ThingsBoard
# alarm rules used by Code/edge_simulation.py)
WARN_THRESHOLD = 50
CRIT_THRESHOLD = 30
CLASSES = ["Healthy", "Warning", "Critical"]

# Training hyperparameters -- identical to Code/train_model.py
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VAL_FRACTION = 0.1
N_REP_SAMPLES = 100          # representative-dataset size for INT8 calibration

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "Code", "CMaps")


# --------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------
def load_raw(data_dir=DATA_DIR, subset="FD001"):
    """Load the C-MAPSS train split, test split and ground-truth test RUL."""
    df_train = pd.read_csv(os.path.join(data_dir, f"train_{subset}.txt"),
                           sep=r"\s+", header=None, names=COLS)
    df_test = pd.read_csv(os.path.join(data_dir, f"test_{subset}.txt"),
                          sep=r"\s+", header=None, names=COLS)
    y_test_true = pd.read_csv(os.path.join(data_dir, f"RUL_{subset}.txt"),
                              sep=r"\s+", header=None, names=["RUL"])
    return df_train, df_test, y_test_true


def add_rul(df, cap=RUL_CAP):
    """Piecewise-linear RUL target: min(max_cycle - cycle, cap)."""
    df = df.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def fit_scaler(df_train, features=FEATURES):
    return df_train[features].min(), df_train[features].max()


def apply_scale(df, mn, mx, features=FEATURES):
    df = df.copy()
    df[features] = (df[features] - mn) / (mx - mn)
    return df


def make_sequences(df, features=FEATURES, seq_len=SEQ_LEN):
    """Sliding windows over each engine's trajectory, with unit ids as groups."""
    X, y, groups = [], [], []
    for u in df["unit"].unique():
        d = df[df["unit"] == u]
        Xu, yu = d[features].values, d["RUL"].values
        for i in range(len(d) - seq_len + 1):
            X.append(Xu[i:i + seq_len])
            y.append(yu[i + seq_len - 1])
            groups.append(u)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(groups))


def make_test_windows(df_test_scaled, features=FEATURES, seq_len=SEQ_LEN):
    """Last `seq_len` cycles per test engine, edge-padded if the engine is short."""
    X = []
    for u in df_test_scaled["unit"].unique():
        d = df_test_scaled[df_test_scaled["unit"] == u]
        w = d[features].values[-seq_len:]
        if len(w) < seq_len:
            w = np.pad(w, ((seq_len - len(w), 0), (0, 0)), mode="edge")
        X.append(w)
    return np.array(X, dtype=np.float32)


def build_dataset(seed, data_dir=DATA_DIR, subset="FD001"):
    """Full preprocessing pipeline. Returns everything the study needs."""
    df_train_raw, df_test_raw, y_test_true_raw = load_raw(data_dir, subset)

    df_train = add_rul(df_train_raw)
    mn, mx = fit_scaler(df_train)
    df_train_s = apply_scale(df_train, mn, mx)

    X_seq, y_seq, groups = make_sequences(df_train_s)

    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=seed)
    tr_idx, val_idx = next(gss.split(X_seq, y_seq, groups=groups))

    df_test_s = apply_scale(df_test_raw, mn, mx)
    X_test = make_test_windows(df_test_s)
    y_true = y_test_true_raw["RUL"].clip(upper=RUL_CAP).values.astype(np.float64)

    return dict(
        X_train=X_seq[tr_idx], y_train=y_seq[tr_idx],
        X_val=X_seq[val_idx], y_val=y_seq[val_idx],
        X_test=X_test, y_true=y_true,
        scaler_min=mn, scaler_max=mx,
        df_test_scaled=df_test_s,
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def cmapss_score(y_true, y_pred):
    """Standard asymmetric C-MAPSS score (Eq. 18). Lower is better.

    Late predictions (y_pred > y_true, i.e. the model thinks the engine has
    more life left than it does) are penalized with exp(d/10); early
    predictions with the gentler exp(-d/13). This is the metric that actually
    encodes 'a late maintenance call is a safety problem, an early one is
    only a cost problem'.
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(d < 0,
                                 np.exp(-d / 13.0) - 1.0,
                                 np.exp(d / 10.0) - 1.0)))


def _skew(x):
    """Fisher-Pearson sample skewness, computed directly (no scipy dependency)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan")
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0:
        return 0.0
    g1 = np.mean(((x - mu) / sd) ** 3)
    return float(math.sqrt(n * (n - 1)) / (n - 2) * g1)


def status_of(rul):
    """Map continuous RUL onto the operational alert classes (Eq. 17)."""
    rul = np.asarray(rul)
    return np.where(rul > WARN_THRESHOLD, "Healthy",
                    np.where(rul > CRIT_THRESHOLD, "Warning", "Critical"))


def confusion(y_true, y_pred):
    """3x3 health-status confusion matrix; rows = true, cols = predicted."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(status_of(y_true), status_of(y_pred)):
        cm[idx[t], idx[p]] += 1
    return cm


def prf_from_cm(cm):
    prec, rec, f1 = [], [], []
    for k in range(3):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        prec.append(float(p))
        rec.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if (p + r) else 0.0)
    return prec, rec, f1


def evaluate(y_true, y_pred):
    """All metrics for one set of predictions.

    Beyond the usual RMSE/MAE this returns the *direction* of error, which is
    the axis the study is built around. Positive residual = the model predicts
    MORE remaining life than the engine actually has = the unsafe direction.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = y_pred - y_true

    cm = confusion(y_true, y_pred)
    prec, rec, f1 = prf_from_cm(cm)
    over = resid[resid > 0]

    return dict(
        # --- conventional accuracy ---
        rmse=float(math.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        # --- safety-asymmetric ---
        cmapss_score=cmapss_score(y_true, y_pred),
        mean_error=float(resid.mean()),          # >0 => optimistic => unsafe
        median_error=float(np.median(resid)),
        residual_skew=_skew(resid),
        late_fraction=float((resid > 0).mean()),  # share of engines over-estimated
        p90_overestimate=float(np.percentile(over, 90)) if over.size else 0.0,
        max_overestimate=float(over.max()) if over.size else 0.0,
        # --- operational / alerting ---
        confusion_matrix=cm.tolist(),
        status_accuracy=float(np.trace(cm) / cm.sum()),
        precision=prec, recall=rec, f1=f1,
        # the single most safety-critical cell: a failing engine called healthy
        critical_called_healthy=int(cm[2, 0]),
        critical_missed=int(cm[2, 0] + cm[2, 1]),
        n_engines=int(len(y_true)),
    )


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------
def _representative_gen_factory(X_train, rng, n=N_REP_SAMPLES):
    """Calibration set: n randomly drawn REAL training windows, so the
    converter sees a genuine mix of healthy and failing engine states."""
    idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)

    def gen():
        for i in idx:
            yield [X_train[i:i + 1].astype(np.float32)]
    return gen


def convert_float32(model):
    """Baseline TFLite conversion with no quantization at all.

    Used so that FP32 and INT8 size/latency are measured inside the SAME
    runtime -- comparing a Keras .predict() call against a TFLite interpreter
    would measure Python overhead, not the model.
    """
    import tensorflow as tf
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    return c.convert()


def convert_int8(model, X_train, rng):
    """Full static INT8 post-training quantization: int8 weights, int8
    activations, int8 input and output tensors. This is the TFLite-Micro
    deployable form -- NOT dynamic-range quantization, which leaves
    activations in float and is far gentler on recurrent models."""
    import tensorflow as tf
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = _representative_gen_factory(X_train, rng)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def tflite_predict(tflite_bytes, X, cap=RUL_CAP):
    """Run every window in X through a TFLite interpreter, one at a time,
    exactly as the edge device does. Handles both float32 and int8 IO by
    reading the interpreter's own quantization parameters."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    in_d, out_d = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]
    out_scale, out_zp = out_d["quantization"]
    is_int8_in = in_d["dtype"] == np.int8
    is_int8_out = out_d["dtype"] == np.int8

    preds = []
    for row in X.astype(np.float32):
        if is_int8_in:
            q = np.clip(np.round(row / in_scale) + in_zp, -128, 127).astype(np.int8)
        else:
            q = row
        it.set_tensor(in_d["index"], q[np.newaxis, ...])
        it.invoke()
        raw = it.get_tensor(out_d["index"])
        if is_int8_out:
            preds.append(out_scale * (int(raw[0][0]) - out_zp))
        else:
            preds.append(float(raw[0][0]))

    # Clamp to the trained target range, matching edge_simulation.py
    return np.clip(np.array(preds, dtype=np.float64), 0.0, cap)


def measure_latency(tflite_bytes, X, n_warmup=20, n_runs=200):
    """Median single-window inference latency in milliseconds, measured on
    this machine's CPU. Reported as a relative comparison between models in
    one identical runtime, NOT as a claim about microcontroller performance."""
    import time
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    in_d = it.get_input_details()[0]
    sample = X[:1].astype(np.float32)
    if in_d["dtype"] == np.int8:
        s, z = in_d["quantization"]
        sample = np.clip(np.round(sample / s) + z, -128, 127).astype(np.int8)

    for _ in range(n_warmup):
        it.set_tensor(in_d["index"], sample)
        it.invoke()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        it.set_tensor(in_d["index"], sample)
        it.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))
