# AeroGuard — Enhanced Study

**Architecture × quantization for turbofan RUL, scored with safety-asymmetric metrics.**

This folder extends the original `Code/` project from a single 1D-CNN into a
controlled four-architecture comparison under **full static INT8** quantization,
and scores it on the axis the existing literature does not report: **which
direction** quantization moves the error.

Nothing here is transcribed, estimated, or carried over. Every number in
`results/` comes from a model that was trained by `run_study.py` and an
interpreter that was invoked on all 100 FD001 test engines in the same run.

---

## 1. What is being compared

Four architecture families, all consuming the identical `(30, 14)` window and
emitting one linear RUL scalar:

| Key | Family | Why it is in the study |
|---|---|---|
| `cnn` | 1D-CNN | The incumbent. Same topology as `Code/train_model.py`, so the study reproduces the original result rather than replacing it. |
| `lstm` | Stacked LSTM | Recurrent. The family with the strongest a-priori reason to degrade: activation quantization error is re-injected into the hidden state at each of the 30 timesteps instead of passing once through a fixed-depth stack. |
| `transformer` | Encoder block (MHA + LayerNorm) | Attention. Softmax and LayerNormalization are the ops usually named as hostile to per-tensor INT8. |
| `mlp` | Flatten + dense | Control. Discards temporal structure entirely; the easiest possible case for post-training quantization, and therefore the floor for "how little can quantization cost". |

Everything else is held fixed and inherited from the original project: RUL
capped at 130, the same 14 retained sensors, Min-Max bounds fit on train only,
`L = 30`, `GroupShuffleSplit(test_size=0.1)` grouped by engine unit, Adam @
1e-3, batch 64, ≤50 epochs, `EarlyStopping(patience=5, restore_best_weights)`,
and a 100-window representative calibration set drawn from real training data.
**The architecture is the only independent variable.**

Each configuration is run at **3 seeds** (42, 1337, 2026). Seeds vary both the
weight initialisation and the engine-level validation split, and every reported
value is a mean ± s.d. over those seeds — a single-run difference between two
architectures is not a result.

---

## 2. The measurement that makes this different

Published INT8 results for C-MAPSS report the cost of quantization as one
scalar, computed with RMSE. RMSE is symmetric: it cannot distinguish predicting
15 cycles *too many* from 15 cycles *too few*. For RUL those are not equivalent
— over-estimating remaining life delays maintenance on a failing engine, while
under-estimating it only triggers maintenance early. So "quantization preserved
accuracy" is a claim about magnitude that is silent about safety.

This study therefore reports, per architecture and per regime:

**Conventional** — `rmse`, `mae`

**Safety-asymmetric** (the contribution)
- `mean_error` — mean signed residual $\hat{RUL} - RUL$. **Positive = optimistic = unsafe.**
- `late_fraction` — share of test engines whose remaining life is over-estimated
- `p90_overestimate`, `max_overestimate` — the size of the unsafe tail
- `residual_skew` — asymmetry of the error distribution
- `cmapss_score` — the standard asymmetric score, which penalises late predictions with `exp(d/10)` against `exp(-d/13)` for early ones

**Operational** — the three-class Healthy / Warning / Critical confusion at the
same thresholds `Code/edge_simulation.py` uses for ThingsBoard alarms, plus
`critical_called_healthy`: a failing engine reported as healthy, the single
cell where a maintenance system actually hurts someone.

**Deployment** — flatbuffer size FP32 vs INT8, realised compression ratio,
median and p95 single-window latency, parameter count, epochs to converge.

The headline quantity is the **delta**: `INT8 − FP32` on each of the above,
paired within seed.

---

## 3. Full static INT8, not dynamic

This matters for comparability. The closest published work
([*Machines* 14(4):375, 2026](https://doi.org/10.3390/machines14040375)) uses
**dynamic-range** INT8 — weights are quantized, activations stay float at
runtime. That sidesteps the exact mechanism that should hurt recurrent models,
and it is also why that paper observes latency *increasing* (per-batch
quantize/dequantize overhead) rather than decreasing.

`common.convert_int8()` targets `TFLITE_BUILTINS_INT8` with `int8` input and
output tensors — weights, activations and IO all integer, the form TFLite-Micro
actually deploys. Results from the dynamic regime do not predict results here.

### A toolchain finding, recorded rather than worked around

With Keras 3 / TF 2.20, an LSTM with the default `unroll=False` **cannot be
converted at all**:

```
error: 'tf.TensorListReserve' op requires element_shape to be static
       during TF Lite transformation pass
```

and the `SELECT_TF_OPS` fallback yields a flatbuffer the standard interpreter
refuses to run without the Flex delegate — which does not exist on
microcontroller targets. Unrolling the 30 timesteps is what makes a fully-INT8
recurrent model convertible, at a real cost in flatbuffer size (the cell graph
is duplicated 30×). `models.py` documents this and `run_study.py` records both
the requirement and its size cost. This is a deployment constraint on the
recurrent family, not a modelling preference.

---

## 4. Layout

```
Code_Enhanced/
├── common.py           data pipeline, metrics, quantization, latency harness
├── models.py           the four architectures
├── run_study.py        trains and evaluates everything -> results/results.json
├── merge_results.py    combines several run_study.py outputs (extra seeds)
├── make_figures.py     renders results/figures/*.png and results/tables.md
├── FINDINGS.md         what the measurements actually showed
└── results/
    ├── results.json    merged: every run's metrics, history, per-engine predictions
    ├── results_a.json  raw output, seeds 42 / 1337 / 2026
    ├── results_b.json  raw output, seeds 7 / 99 / 2024
    ├── summary.json    mean/std per architecture per regime
    ├── tables.md       Tables E1-E5
    ├── run_log.txt     console transcript, first batch
    ├── run_log_b.txt   console transcript, second batch
    ├── figures/        e1 ... e9
    └── artifacts/      the actual .tflite files (first seed of each batch)
```

The study was run in two batches of 3 seeds and merged with `merge_results.py`,
which refuses to combine files whose experimental setup differs. Both raw
outputs are kept so the merge is auditable.

`results.json` stores the **per-engine predictions** for every run, so every
figure and table can be recomputed from it without retraining, and any number
quoted from this study can be traced back to the array it came from.

---

## 5. Running it

```bash
cd Code_Enhanced && python run_study.py
```

```bash
cd Code_Enhanced && python make_figures.py
```

`python run_study.py --smoke` does a 1-seed, 2-epoch pass to verify the
pipeline. `--archs cnn lstm` and `--seeds 42 1337` subset the study.

Requires `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`.
Developed on TensorFlow 2.20 / Python 3.13, CPU-only. Reads the dataset from
`../Code/CMaps/`.

---

## 6. Figures

| File | Shows |
|---|---|
| `e1_accuracy_fp32_vs_int8.png` | RMSE and MAE per architecture, both regimes, error bars over seeds |
| `e2_safety_bias_shift.png` | **Headline.** FP32→INT8 shift in signed residual, and how often each model is optimistic |
| `e3_residual_distributions.png` | Residual histograms, FP32 vs INT8 overlaid, unsafe tail to the right |
| `e4_asymmetric_score.png` | Safety-weighted C-MAPSS score, log scale |
| `e5_deployment_tradeoff.png` | INT8 size vs RMSE vs latency, and realised compression |
| `e6_confusion_grid.png` | INT8 health-status confusion, failing-called-healthy cell boxed |
| `e7_delta_heatmap.png` | Quantization delta across every metric at once |
| `e8_loss_curves.png` | Convergence by family |
| `e9_per_seed_deltas.png` | Every seed plotted individually, so divergent runs stay visible rather than hidden in a mean |

---

## 7. Limitations, stated up front

- **No microcontroller.** Latency and size are measured in the TFLite
  interpreter on a laptop CPU. They are a like-for-like comparison *between
  architectures in one runtime*, not a claim about MCU performance. An
  on-device measurement is the single largest strengthener available, and the
  bar in this niche is now visibly real hardware.
- **FD001 only**, inherited from the original project. The multi-regime subsets
  (FD002/FD004) widen the dynamic range that calibration has to cover and are
  where quantization stress should be most visible.
- **3 seeds** is enough to stop single-run noise being mistaken for a result;
  it is not enough for a significance claim on small deltas.
- Architectures are matched on the training protocol and roughly on parameter
  budget, not tuned per family. A tuned LSTM might do better; this study
  measures the families as a practitioner would first reach for them.
- The status-class reframing inherits the original project's weakness: the
  Warning band is 20 cycles wide against a model whose RMSE is ~15, so a share
  of Warning engines will always spill into neighbouring classes from ordinary
  regression noise.

---

## 8. Relationship to prior work

| Prior work | What it establishes | What it leaves open |
|---|---|---|
| [*Machines* 14(4):375, 2026](https://doi.org/10.3390/machines14040375) — CPU deployment-oriented evaluation, CNN–GRU vs CNN–TCN | Compact architectures on C-MAPSS under **dynamic** INT8, latency on x86 and ARM | Only two hybrid architectures; dynamic quantization; accuracy reported symmetrically |
| ["From Deep Learning to Microcontroller" — KD + INT8 on ESP32-S3](https://ieee-dataport.org/documents/source-code-models-and-results-knowledge-distillation-and-int8-quantization-turbofan-rul) | 9-architecture baseline on FD001, INT8 QAT, real MCU profiling; deployment feasibility as the binding constraint | Cost of quantization reported as a single accuracy scalar |
| [SARNet, 2025](https://arxiv.org/pdf/2510.22955) | Prediction jitter is real; consecutive-spike validation reduces false alarms | Not compared across architectures, and not before/after quantization |
| [QuantiBias, 2026](https://arxiv.org/abs/2607.21063) | Quantization induces directional bias that standard evaluation misses — in LLMs | Never connected to prognostics, where the direction has a safety meaning |

The gap this study occupies is the intersection: **full static INT8 × multiple
architecture families × safety-asymmetric scoring.** The individual ingredients
exist; the claim that "INT8 is accuracy-neutral for RUL" has not been checked
with a metric capable of detecting a directional shift.

See `results/tables.md` and `FINDINGS.md` for what the measurements actually
showed — including where the expected effect did *not* appear.
