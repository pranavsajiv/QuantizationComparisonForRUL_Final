"""
Ablation study and hyperparameter sweep for the enhanced AeroGuard paper.

Produces the real, multi-seed numbers behind Sections 4.5 and 4.6. Every
variant is trained from scratch and evaluated in BOTH numeric regimes, so an
ablation can be read for its effect on the deployed INT8 model rather than
only on the float one -- which is the whole point of the study.

ABLATIONS (Section 4.5), run for the CNN and the LSTM at 3 seeds each
    full             the proposed configuration
    no_rul_cap       raw linear RUL target, no piecewise cap at 130
    all_21_sensors   Module 2 feature pruning disabled
    short_window     L = 15 instead of 30

HYPERPARAMETER SWEEPS (Section 4.6)
    window length    L in {15, 20, 25, 30, 40, 50}, both architectures, 3 seeds
    calibration size n in {10, 50, 100, 500} representative windows -- swept on
                     each already-trained L=30 model, so it isolates the effect
                     of the calibration set alone with the weights held fixed

USAGE
    python run_ablations.py                 # everything
    python run_ablations.py --skip-sweep    # ablations only
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import time
import numpy as np
import tensorflow as tf

import common as C
from models import BUILDERS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

SEEDS = [42, 1337, 2026]
ARCHS = ["cnn", "lstm"]
ALL_SENSORS = [f"s{i}" for i in range(1, 22)]
CALIB_SIZES = [10, 50, 100, 500]
WINDOW_LENGTHS = [15, 20, 25, 30, 40, 50]


def _log(m):
    print(m, flush=True)


def zero_range_sensors(features):
    """Sensors whose FD001 training range is exactly zero. Min-Max scaling
    divides by that range, so including them without a guard is a literal
    0/0 -- the reason Module 2's variance mask is a correctness requirement
    and not only a memory optimisation."""
    df_train_raw, _, _ = C.load_raw()
    rng = df_train_raw[features].max() - df_train_raw[features].min()
    return [f for f in features if rng[f] == 0]


def prepare(seed, features, seq_len, cap, guard_scaler=True):
    """Preprocessing with any single choice overridden.

    guard_scaler=False reproduces the naive pipeline exactly: the Min-Max
    denominator is used as-is, so a constant sensor produces NaN.
    """
    df_train_raw, df_test_raw, y_test_true_raw = C.load_raw()

    df_train = df_train_raw.copy()
    df_train["RUL"] = (df_train.groupby("unit")["cycle"].transform("max")
                       - df_train["cycle"])
    if cap is not None:
        df_train["RUL"] = df_train["RUL"].clip(upper=cap)

    mn, mx = C.fit_scaler(df_train, features)
    if guard_scaler:
        # constant channels collapse to 0 instead of NaN
        mx = mx.copy()
        mx[(mx - mn) == 0] = mn[(mx - mn) == 0] + 1.0
    df_train_s = C.apply_scale(df_train, mn, mx, features)
    X_seq, y_seq, groups = C.make_sequences(df_train_s, features, seq_len)

    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=C.VAL_FRACTION,
                            random_state=seed)
    tr, va = next(gss.split(X_seq, y_seq, groups=groups))

    df_test_s = C.apply_scale(df_test_raw, mn, mx, features)
    X_test = C.make_test_windows(df_test_s, features, seq_len)
    y_true = y_test_true_raw["RUL"]
    y_true = (y_true.clip(upper=cap) if cap is not None else y_true)
    return (X_seq[tr], y_seq[tr], X_seq[va], y_seq[va],
            X_test, y_true.values.astype(np.float64))


def train_variant(arch, seed, features, seq_len, cap, calib_sizes, tag,
                  guard_scaler=True):
    """Train one variant, evaluate FP32 and INT8 at each calibration size."""
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    X_tr, y_tr, X_val, y_val, X_test, y_true = prepare(
        seed, features, seq_len, cap, guard_scaler=guard_scaler)

    # A naive scaler over constant sensors produces NaN before training ever
    # starts. Record that as the measured outcome instead of crashing.
    n_bad = int(np.isnan(X_tr).sum() + np.isnan(X_test).sum())
    if n_bad:
        dead = zero_range_sensors(features)
        _log(f"  [{tag:22s} {arch:4s} s{seed:<5d}] PREPROCESSING FAILED -- "
             f"{n_bad:,} NaN inputs from {len(dead)} zero-range sensors "
             f"({', '.join(dead)})")
        return dict(tag=tag, arch=arch, seed=seed, seq_len=seq_len,
                    rul_cap=cap, n_features=len(features),
                    preprocessing_failed=True, nan_inputs=n_bad,
                    zero_range_sensors=dead, fp32=None,
                    int8_by_calibration={})

    model = BUILDERS[arch](seq_len=seq_len, n_features=len(features))
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=C.PATIENCE, restore_best_weights=True)
    t0 = time.perf_counter()
    hist = model.fit(X_tr, y_tr, epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
                     validation_data=(X_val, y_val), callbacks=[es], verbose=0)
    secs = time.perf_counter() - t0

    clip_cap = cap if cap is not None else float(y_true.max())
    fp32_bytes = C.convert_float32(model)
    y_fp32 = C.tflite_predict(fp32_bytes, X_test, cap=clip_cap)
    m_fp32 = C.evaluate(y_true, y_fp32)

    rec = dict(tag=tag, arch=arch, seed=seed, seq_len=seq_len,
               rul_cap=cap, n_features=len(features),
               preprocessing_failed=False, guard_scaler=guard_scaler,
               epochs_run=len(hist.history["loss"]),
               train_seconds=round(secs, 1),
               fp32_size_bytes=len(fp32_bytes),
               fp32={k: v for k, v in m_fp32.items()
                     if k not in ("confusion_matrix",)},
               int8_by_calibration={})

    for n in calib_sizes:
        try:
            b = C.convert_int8(model, X_tr, np.random.default_rng(seed),
                               ) if n == C.N_REP_SAMPLES else None
            if b is None:
                # explicit calibration size, same generator contract
                conv = tf.lite.TFLiteConverter.from_keras_model(model)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.representative_dataset = C._representative_gen_factory(
                    X_tr, np.random.default_rng(seed), n=n)
                conv.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type = tf.int8
                conv.inference_output_type = tf.int8
                b = conv.convert()
            y_q = C.tflite_predict(b, X_test, cap=clip_cap)
            m_q = C.evaluate(y_true, y_q)
            rec["int8_by_calibration"][str(n)] = dict(
                {k: v for k, v in m_q.items() if k not in ("confusion_matrix",)},
                size_bytes=len(b), converted=True)
        except Exception as e:
            rec["int8_by_calibration"][str(n)] = dict(
                converted=False, error=f"{type(e).__name__}: {str(e)[:300]}")

    base = rec["int8_by_calibration"].get(str(C.N_REP_SAMPLES), {})
    _log(f"  [{tag:22s} {arch:4s} s{seed:<5d}] "
         f"FP32 RMSE {m_fp32['rmse']:6.2f} bias {m_fp32['mean_error']:+6.2f} | "
         f"INT8 RMSE {base.get('rmse', float('nan')):6.2f} "
         f"bias {base.get('mean_error', float('nan')):+6.2f} | "
         f"{rec['epochs_run']:2d} ep {secs:5.0f}s")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--archs", nargs="+", default=ARCHS)
    ap.add_argument("--out", default="ablations.json")
    args = ap.parse_args()
    archs = args.archs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = dict(meta=dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tensorflow=tf.__version__, seeds=SEEDS, archs=archs,
        calibration_sizes=CALIB_SIZES, window_lengths=WINDOW_LENGTHS,
    ), ablations=[], sweep_window=[], sweep_calibration=[])

    t_start = time.perf_counter()
    path = os.path.join(RESULTS_DIR, args.out)

    def save(o):
        o["meta"]["total_seconds"] = round(time.perf_counter() - t_start, 1)
        with open(path, "w") as fh:
            json.dump(o, fh, indent=2)

    if not args.skip_ablation:
        _log("\n=== ABLATIONS (Section 4.5) ===")
        variants = [
            ("full", C.FEATURES, 30, C.RUL_CAP, True),
            ("no_rul_cap", C.FEATURES, 30, None, True),
            ("all_21_naive", ALL_SENSORS, 30, C.RUL_CAP, False),
            ("all_21_guarded", ALL_SENSORS, 30, C.RUL_CAP, True),
            ("short_window_L15", C.FEATURES, 15, C.RUL_CAP, True),
        ]
        for tag, feats, L, cap, guard in variants:
            for arch in archs:
                for seed in SEEDS:
                    out["ablations"].append(
                        train_variant(arch, seed, feats, L, cap,
                                      [C.N_REP_SAMPLES], tag,
                                      guard_scaler=guard))
                    save(out)

    if not args.skip_sweep:
        _log("\n=== WINDOW-LENGTH SWEEP (Section 4.6) ===")
        for L in WINDOW_LENGTHS:
            for arch in archs:
                for seed in SEEDS:
                    calib = CALIB_SIZES if L == 30 else [C.N_REP_SAMPLES]
                    rec = train_variant(arch, seed, C.FEATURES, L, C.RUL_CAP,
                                        calib, f"L={L}")
                    out["sweep_window"].append(rec)
                    if L == 30:
                        out["sweep_calibration"].append(rec)
                    save(out)

    save(out)
    _log(f"\nWrote {path} "
         f"({len(out['ablations'])} ablation runs, "
         f"{len(out['sweep_window'])} sweep runs, "
         f"{out['meta']['total_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
