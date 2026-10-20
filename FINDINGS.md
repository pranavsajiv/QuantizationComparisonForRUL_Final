# Findings

4 architectures × 6 seeds × {FP32, full-static-INT8} = 24 trained models,
each evaluated on all 100 C-MAPSS FD001 test engines.
Every number below is traceable to `results/results.json`; per-seed values are
in `results/tables.md` Table E4 with **no runs excluded**.

---

## 1. Static INT8 is not accuracy-neutral — but only for one family

| Architecture | FP32 RMSE | INT8 RMSE | Δ RMSE | Seeds worse |
|---|---|---|---|---|
| 1D-CNN | 14.63 ± 0.31 | 14.66 ± 0.32 | **+0.03** | 3/6 |
| **LSTM** | 18.92 ± 9.90 | 21.71 ± 8.90 | **+2.79** | **5/6** |
| Transformer | 16.45 ± 1.29 | 16.54 ± 1.34 | **+0.09** | 4/6 |
| MLP | 14.80 ± 0.42 | 14.79 ± 0.36 | **−0.01** | 3/6 |

The CNN, Transformer and MLP all quantize for free — deltas are inside seed
noise, and the sign is a coin flip. The LSTM does not: a mean **+2.79 cycle**
penalty, reaching **+7.71** at one seed.

Excluding the single divergent LSTM run (seed 1337, see §5), the effect is
cleaner still: **14.50 ± 0.74 → 17.86 ± 2.50**. Quantization does not merely
shift the recurrent model's error up, it **triples the spread** — the INT8
LSTM is both worse and far less predictable than the model it came from.

This is the mechanism the study was built to test. In a feed-forward stack
activation quantization error passes through a fixed number of layers once. In
an unrolled recurrent cell it is re-injected into the hidden state at each of
the 30 timesteps.

---

## 2. Whenever the LSTM won in FP32, it lost after quantization

| Seed | Best FP32 | Best INT8 | |
|---|---|---|---|
| 42 | **lstm** (13.69) | mlp (14.85) | inverted |
| 7 | **lstm** (13.69) | cnn (14.70) | inverted |
| 1337 | cnn (14.17) | cnn (14.14) | — |
| 2026 | cnn (14.75) | cnn (14.69) | — |
| 99 | mlp (14.18) | mlp (14.17) | — |
| 2024 | cnn (14.39) | cnn (14.41) | — |

The LSTM was the most accurate FP32 model at two of six seeds, and at **both**
of those it fell behind after quantization. The ranking measured in float does
not survive to the deployed artefact. A study that stops at FP32 RMSE — which
is most of the C-MAPSS literature — would have selected the LSTM.

---

## 3. Two deployment results that invert the usual assumption

`results/tables.md` Table E3, deterministic (not seed-dependent):

- **The LSTM's INT8 flatbuffer is *larger* than its FP32 one: 305.4 KB vs
  290.0 KB — a compression ratio of 0.95×.** Unrolling the 30 timesteps, which
  is what makes a fully-INT8 recurrent model convertible at all (§4),
  duplicates the cell graph 30 times; the per-tensor quantization metadata that
  adds outweighs the 4:1 saving on weights. Against the CNN's 2.86× and the
  MLP's 3.43×, the recurrent family pays for quantization and receives nothing.
- **The Transformer's INT8 model is 11.6× *slower* than its FP32 model:
  0.116 ms vs 0.010 ms per window.** The CNN and MLP both get faster
  (0.005→0.003 ms, 0.003→0.002 ms). Quantized attention is not a speed win in
  this runtime.

"INT8 makes models smaller and faster" holds for two of the four families
tested and fails for the other two, in opposite ways.

---

## 4. A recurrent model cannot be fully INT8-quantized at all by default

With Keras 3 / TF 2.20, an LSTM with the default `unroll=False` fails
conversion outright:

```
error: 'tf.TensorListReserve' op requires element_shape to be static
       during TF Lite transformation pass
```

The `SELECT_TF_OPS` fallback produces a flatbuffer the standard interpreter
refuses to execute without the Flex delegate, which does not exist on
microcontroller targets. `unroll=True` is what makes the conversion possible —
and is also the direct cause of the size inversion in §3. This was measured,
not assumed; see the probe results reproduced in `models.py`.

---

## 5. The safety-direction hypothesis: partly confirmed, and weaker than expected

This is where the study did **not** return what it was set up to find, and the
distinction matters.

**Not confirmed — there is no unconditional directional bias.** Counting how
many seeds moved toward over-estimating remaining life:

| Architecture | Δ bias > 0 | mean Δ bias |
|---|---|---|
| 1D-CNN | 1/6 | −0.08 |
| LSTM | 3/6 | +1.91 |
| Transformer | 5/6 | +0.18 |
| MLP | 4/6 | −0.01 |

The LSTM is a coin flip on sign. The CNN actually drifts slightly *safer*. The
Transformer is the most consistent (5/6) but the magnitude is negligible.
**"INT8 systematically makes RUL models optimistic" is not supported.**

**Confirmed, conditionally — when quantization does damage, the damage is
optimistic.** Correlation between per-seed Δ RMSE and Δ signed bias:

| Architecture | r(ΔRMSE, Δbias) | mean positive Δbias | mean negative Δbias |
|---|---|---|---|
| 1D-CNN | +0.24 | +0.50 | −0.20 |
| **LSTM** | **+0.87** | **+4.84** | −1.02 |
| Transformer | +0.77 | +0.34 | −0.63 |
| MLP | +0.91 | +0.24 | −0.51 |

For the LSTM the three seeds with the largest accuracy loss are exactly the
three whose bias moved toward over-estimation (+7.18, +4.09, +3.25), while the
seeds that moved the safe way barely moved at all (−2.24, −0.63, −0.19). When
it goes wrong it goes wrong by roughly **5× further in the unsafe direction
than in the safe one**.

The operational consequence, summed over 6 seeds — engines truly in the
Critical band that INT8 failed to flag as Critical:

| Architecture | FP32 | INT8 |
|---|---|---|
| 1D-CNN | 4 | 3 |
| **LSTM** | **35** | **47** |
| Transformer | 27 | 29 |
| MLP | 21 | 22 |

Quantizing the LSTM produced **12 additional missed Critical engines**. No
architecture ever reported a truly-Critical engine as fully Healthy except the
LSTM (25 of them) and the Transformer (4); the CNN and MLP never did.

**The defensible claim is therefore narrower than the one the study set out to
test:** quantization damage is not universal and not universally directional,
but it is architecture-dependent, and where it occurs it is asymmetric toward
the unsafe side — which a symmetric RMSE delta cannot report.

---

## 6. LSTM training is unstable on this dataset

FP32 RMSE standard deviation across 6 seeds: CNN **0.31**, MLP **0.42**,
Transformer **1.29**, LSTM **9.90**. Seed 1337 stopped after 7 epochs at RMSE
41.01 — a genuine non-convergence, retained in all tables rather than dropped.
This is an independent reason to be cautious about single-seed architecture
comparisons on C-MAPSS, which is how most of them are reported.

---

## 7. The ablations: feature pruning is a correctness requirement, not an accuracy one

Five variants, 1D-CNN and LSTM, 3 seeds each, every one scored in both regimes
(`results/ablations.json`, Table 5 of the paper).

| Variant | CNN FP32 | CNN INT8 | LSTM FP32 | LSTM INT8 |
|---|---|---|---|---|
| Proposed (full system) | 14.56 ± 0.28 | 14.58 ± 0.33 | 23.45 ± 12.44 | 25.85 ± 10.77 |
| No piecewise RUL cap | 21.78 ± 1.38 | 21.84 ± 1.37 | 23.12 ± 1.57 | 24.69 ± 1.47 |
| All 21 sensors, **naive** scaler | **FAILS** | **FAILS** | **FAILS** | **FAILS** |
| All 21 sensors, guarded scaler | 14.85 ± 0.44 | 14.87 ± 0.45 | 23.44 ± 12.46 | 26.65 ± 10.52 |
| Shorter window, L = 15 | 19.23 ± 0.66 | 19.25 ± 0.59 | 34.06 ± 11.28 | 34.22 ± 10.88 |

- **The piecewise RUL cap remains the single largest design decision**: removing
  it costs the CNN **+7.2 RMSE** (14.56 → 21.78), consistently across seeds.
- **Feature pruning corrects a divide-by-zero, it does not buy accuracy.** Six
  FD001 sensors (s1, s5, s10, s16, s18, s19) have *exactly* zero training range,
  so Eq. (4) is 0/0 and the unguarded pipeline emits ~2.9 million NaN inputs and
  cannot train at all. Guard the denominator and the seven extra channels cost
  the CNN **+0.29 RMSE** — near-free. This supersedes the "RMSE collapses to
  74.7" figure in the Review-I report; NaN propagates, it does not degrade
  gracefully, so that number could not have come from this pipeline.
- **The architecture effect survives every ablation.** The CNN's quantisation
  penalty is +0.02 in all four runnable variants; the LSTM's is +2.40, +1.57,
  +3.22, +0.16. It is a property of the family, not of one configuration.

## 8. The strongest result: the best float model is the worst deployed model

The window-length sweep (L ∈ {15, 20, 25, 30, 40, 50}, both architectures,
3 seeds, both regimes — Table 7 and Fig. E11):

| L | CNN FP32 | CNN INT8 | LSTM FP32 | LSTM INT8 | LSTM Δ |
|---|---|---|---|---|---|
| 15 | 19.23 | 19.25 | 34.06 | 34.22 | +0.16 |
| 20 | 17.73 | 17.86 | **17.63** | **17.89** | +0.26 |
| 25 | 14.74 | 14.67 | 24.09 | 25.10 | +1.02 |
| 30 | 14.56 | 14.58 | 23.45 | 25.85 | +2.40 |
| 40 | **14.31** | **14.27** | **13.85 ± 0.05** | 32.56 ± 3.58 | **+18.71** |
| 50 | 14.91 | 14.91 | 23.22 | 35.97 | +12.75 |

For the CNN both regimes agree — L = 40 is optimal in float and in INT8, so
tuning on the float model selects correctly.

For the LSTM they do not agree at all. **L = 40 produces the lowest and by far
the most stable floating-point result in the entire study (13.85 ± 0.05) and is
simultaneously its worst deployable configuration (32.56 ± 3.58, a +18.71
penalty).** The window length that actually minimises deployed error is L = 20.

Hyperparameters are conventionally selected against float validation error. For
this architecture that convention picks the single worst option in the search
space, and nothing on the float side would reveal it. This is the same
FP32→INT8 inversion as §2, but sharper: it appears *within one architecture*,
across a routine hyperparameter, at a magnitude an order larger.

**And it is not a calibration artefact.** Varying the calibration set fiftyfold
(n ∈ {10, 50, 100, 500}, weights held fixed) moves CNN INT8 RMSE between 14.51
and 14.58, and leaves the LSTM inside its seed spread (25.53–25.85). The first
and cheapest explanation for the recurrent penalty — an under-sized or
unrepresentative calibration set — is ruled out by measurement.

## 9. What is novel here, against the literature actually checked

Prior work establishes the pieces separately:

- [*Machines* 14(4):375, 2026](https://doi.org/10.3390/machines14040375) compares
  CNN–GRU vs CNN–TCN on C-MAPSS under **dynamic** INT8 with x86/ARM latency.
  Dynamic quantization leaves activations in float, which sidesteps the exact
  recurrent accumulation mechanism §1 measures — and explains why that paper
  reports quantization "preserving predictive accuracy" where this one does not.
- ["From Deep Learning to Microcontroller"](https://ieee-dataport.org/documents/source-code-models-and-results-knowledge-distillation-and-int8-quantization-turbofan-rul)
  runs a 9-architecture FD001 baseline with INT8 QAT on a real ESP32-S3 and
  argues deployment feasibility is the binding constraint — but reports the
  cost of quantization as one accuracy scalar.
- [SARNet, 2025](https://arxiv.org/pdf/2510.22955) establishes that RUL
  prediction jitter is real and worth debouncing, without comparing
  architectures or numeric regimes.
- [QuantiBias, 2026](https://arxiv.org/abs/2607.21063) shows quantization
  induces directional bias invisible to standard evaluation — in LLMs, where
  the direction carries no safety meaning.

Four things in this study do not appear in that set:

1. **Full static INT8 across four architecture families on identical data,
   identical training protocol and 6 paired seeds**, with the per-seed values
   published rather than a single run per cell.
2. **The FP32→INT8 ranking inversion demonstrated rather than asserted** — the
   best float model losing to a worse float model after deployment (§2).
3. **Two deployment inversions that contradict the standard framing**: an INT8
   model *larger* than its FP32 source, and an INT8 model 11.6× *slower* (§3).
4. **Quantization cost reported directionally**, with the safety-asymmetric
   result stated at the strength the data supports — conditional, not
   universal (§5). The negative half of that result is itself worth reporting,
   because "INT8 makes RUL models optimistic" is a plausible claim that this
   evidence does not support.
5. **A hyperparameter sweep scored in both regimes rather than in float alone**
   (§8), showing that the window length minimising float error and the one
   minimising deployed error can differ, by +18.71 RMSE, inside a single
   architecture — with the calibration-set explanation ruled out by a
   weights-fixed sweep.

### How strong is this?

Honestly: §1–§4 are solid and reproducible from the artefacts in `results/`.
§5 is a real but conditional effect that needs more seeds to firm up, and its
strongest single number (r = +0.87) rests on 6 points. The whole study is CPU
and FD001 only. The two upgrades that would matter most, in order:

1. **Real microcontroller measurement.** Size and latency here are laptop
   interpreter numbers. The §3 inversions are the kind of claim a reviewer
   will want measured on the target, and the prior work now sets that bar.
2. **FD002/FD004.** Multi-regime data widens the dynamic range calibration has
   to cover, which is precisely where static quantization should hurt more.

A third, cheaper upgrade: 15–20 seeds for the LSTM alone would settle whether
the sign split in §5 is genuinely 50/50 or an artefact of n = 6.
