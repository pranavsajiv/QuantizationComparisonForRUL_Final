"""
AeroGuard enhanced study
========================
Architecture x quantization, scored with safety-asymmetric metrics.

WHAT THIS RUNS
    4 architectures (1D-CNN, LSTM, Transformer, MLP)
  x N seeds (default 3)
  x 2 numeric regimes (FP32 TFLite, full-static-INT8 TFLite)

    ...training every model from scratch and invoking every interpreter on
    every one of the 100 FD001 test engines. No metric in results.json is
    copied, estimated or carried over from the original Code/ run.

WHY IT IS SET UP THIS WAY
    Published INT8 results for C-MAPSS report the cost of quantization as a
    single scalar ("1.6% degradation", "preserves predictive accuracy"),
    computed with RMSE -- a symmetric metric. RMSE cannot see WHICH DIRECTION
    the error moved. In RUL prediction the two directions are not equivalent:
    over-estimating remaining life delays maintenance on a failing engine,
    while under-estimating only triggers it early. This study therefore
    reports, for every architecture, the FP32 -> INT8 shift in

        * mean signed residual   (>0 = optimistic = unsafe)
        * late_fraction          (share of engines whose life is over-estimated)
        * the asymmetric C-MAPSS score
        * critical_called_healthy (a failing engine reported as healthy)

    alongside RMSE/MAE, plus the deployment costs (flatbuffer size, single-
    window latency) that decide whether an architecture is usable at all.

USAGE
    python run_study.py                     # full study, seeds 42 1337 2026
    python run_study.py --smoke             # 1 seed, 2 epochs -- pipeline check
    python run_study.py --seeds 42 1337     # custom seeds
    python run_study.py --archs cnn lstm    # subset of architectures
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import platform
import time
import numpy as np
import tensorflow as tf

import common as C
from models import BUILDERS, ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
ARTIFACT_DIR = os.path.join(RESULTS_DIR, "artifacts")

DEFAULT_SEEDS = [42, 1337, 2026]


def _log(msg):
    print(msg, flush=True)


def run_one(arch, seed, epochs, save_artifacts):
    """Train one architecture at one seed, evaluate FP32 and INT8, return a
    fully self-describing record. Every field is measured, never assumed."""
    _log(f"\n{'='*72}\n  {arch.upper()}  |  seed {seed}\n{'='*72}")

    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    data = C.build_dataset(seed)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_true = data["X_test"], data["y_true"]

    model = BUILDERS[arch]()
    n_params = int(model.count_params())
    _log(f"  params: {n_params:,}   train {X_tr.shape}   val {X_val.shape}")

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=C.PATIENCE, restore_best_weights=True)

    t0 = time.perf_counter()
    hist = model.fit(X_tr, y_tr, epochs=epochs, batch_size=C.BATCH_SIZE,
                     validation_data=(X_val, y_val), callbacks=[es], verbose=0)
    train_secs = time.perf_counter() - t0
    epochs_run = len(hist.history["loss"])
    _log(f"  trained {epochs_run} epochs in {train_secs:.0f}s "
         f"(best val_loss {min(hist.history['val_loss']):.2f})")

    rec = dict(
        arch=arch, seed=seed, n_params=n_params,
        epochs_run=epochs_run, train_seconds=round(train_secs, 1),
        history={k: [float(v) for v in vs] for k, vs in hist.history.items()},
    )

    # ---- Keras reference prediction (cross-check on the TFLite FP32 arm) ----
    y_keras = np.clip(model.predict(X_test, verbose=0).flatten().astype(np.float64),
                      0.0, C.RUL_CAP)
    rec["keras_rmse"] = C.evaluate(y_true, y_keras)["rmse"]

    # ---------------------------- FP32 TFLite -------------------------------
    fp32_bytes = C.convert_float32(model)
    y_fp32 = C.tflite_predict(fp32_bytes, X_test)
    m_fp32 = C.evaluate(y_true, y_fp32)
    lat_fp32, lat_fp32_p95 = C.measure_latency(fp32_bytes, X_test)
    rec["fp32"] = dict(
        **m_fp32,
        size_bytes=len(fp32_bytes),
        latency_ms_median=round(lat_fp32, 4),
        latency_ms_p95=round(lat_fp32_p95, 4),
        predictions=[round(float(v), 4) for v in y_fp32],
    )
    _log(f"  FP32  RMSE {m_fp32['rmse']:6.2f}  MAE {m_fp32['mae']:6.2f}  "
         f"score {m_fp32['cmapss_score']:9.1f}  bias {m_fp32['mean_error']:+6.2f}  "
         f"{len(fp32_bytes)/1024:7.1f} KB  {lat_fp32:.3f} ms")

    # ---------------------------- INT8 TFLite -------------------------------
    try:
        t0 = time.perf_counter()
        int8_bytes = C.convert_int8(model, X_tr, rng)
        convert_secs = time.perf_counter() - t0
        y_int8 = C.tflite_predict(int8_bytes, X_test)
        m_int8 = C.evaluate(y_true, y_int8)
        lat_int8, lat_int8_p95 = C.measure_latency(int8_bytes, X_test)
        rec["int8"] = dict(
            **m_int8,
            size_bytes=len(int8_bytes),
            latency_ms_median=round(lat_int8, 4),
            latency_ms_p95=round(lat_int8_p95, 4),
            convert_seconds=round(convert_secs, 1),
            converted=True,
            predictions=[round(float(v), 4) for v in y_int8],
        )
        _log(f"  INT8  RMSE {m_int8['rmse']:6.2f}  MAE {m_int8['mae']:6.2f}  "
             f"score {m_int8['cmapss_score']:9.1f}  bias {m_int8['mean_error']:+6.2f}  "
             f"{len(int8_bytes)/1024:7.1f} KB  {lat_int8:.3f} ms")
        _log(f"  DELTA RMSE {m_int8['rmse']-m_fp32['rmse']:+6.2f}   "
             f"bias {m_int8['mean_error']-m_fp32['mean_error']:+6.2f}   "
             f"late_frac {m_int8['late_fraction']-m_fp32['late_fraction']:+.3f}   "
             f"compression {len(fp32_bytes)/len(int8_bytes):.2f}x")

        if save_artifacts:
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            with open(os.path.join(ARTIFACT_DIR, f"{arch}_seed{seed}_int8.tflite"), "wb") as fh:
                fh.write(int8_bytes)
            with open(os.path.join(ARTIFACT_DIR, f"{arch}_seed{seed}_fp32.tflite"), "wb") as fh:
                fh.write(fp32_bytes)
    except Exception as e:
        rec["int8"] = dict(converted=False,
                           error=f"{type(e).__name__}: {str(e)[:500]}")
        _log(f"  INT8  CONVERSION FAILED -- {type(e).__name__}: {str(e)[:200]}")

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--archs", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 2 epochs -- verifies the pipeline end to end")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    seeds, epochs = args.seeds, args.epochs
    if args.smoke:
        seeds, epochs = [args.seeds[0]], 2

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ground truth is identical for every run; store it once.
    y_true = C.build_dataset(seeds[0])["y_true"]

    meta = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tensorflow=tf.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        gpu_devices=[d.name for d in tf.config.list_physical_devices("GPU")],
        subset="FD001",
        seq_len=C.SEQ_LEN, rul_cap=C.RUL_CAP, features=C.FEATURES,
        epochs_max=epochs, patience=C.PATIENCE, batch_size=C.BATCH_SIZE,
        learning_rate=C.LEARNING_RATE, calibration_samples=C.N_REP_SAMPLES,
        quantization="full static INT8 (int8 weights, activations, input and output)",
        seeds=seeds, archs=args.archs,
        smoke=args.smoke,
        thresholds=dict(warning=C.WARN_THRESHOLD, critical=C.CRIT_THRESHOLD),
        y_true=[float(v) for v in y_true],
    )

    runs, t_start = [], time.perf_counter()
    for arch in args.archs:
        for seed in seeds:
            runs.append(run_one(arch, seed, epochs,
                                save_artifacts=(seed == seeds[0])))

    meta["total_seconds"] = round(time.perf_counter() - t_start, 1)
    out_path = os.path.join(RESULTS_DIR, args.out)
    with open(out_path, "w") as fh:
        json.dump(dict(meta=meta, runs=runs), fh, indent=2)

    _log(f"\n{'='*72}")
    _log(f"  Wrote {out_path}  ({len(runs)} runs, {meta['total_seconds']:.0f}s total)")
    _log(f"  Next: python make_figures.py")
    _log(f"{'='*72}")


if __name__ == "__main__":
    main()
