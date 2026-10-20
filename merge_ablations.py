"""
Merge several run_ablations.py outputs into one results/ablations.json.

The ablation study was run in two batches so that architectures could be added
without discarding compute already spent. This refuses to merge files whose
experimental setup differs, and refuses to double-count a run.

    python merge_ablations.py ablations_a.json ablations_b.json --out ablations.json
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

CRITICAL = ["seeds", "calibration_sizes", "window_lengths", "tensorflow"]
SECTIONS = ["ablations", "sweep_window", "sweep_calibration"]


def key(r):
    return (r["tag"], r["arch"], r["seed"], r["seq_len"], r.get("rul_cap"),
            r.get("n_features"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="ablations.json")
    args = ap.parse_args()

    merged, archs, seen = None, [], {s: set() for s in SECTIONS}

    for name in args.inputs:
        path = name if os.path.isabs(name) else os.path.join(RESULTS_DIR, name)
        with open(path) as fh:
            d = json.load(fh)

        if merged is None:
            merged = {"meta": dict(d["meta"]),
                      **{s: [] for s in SECTIONS}}
        else:
            for k in CRITICAL:
                if merged["meta"].get(k) != d["meta"].get(k):
                    raise SystemExit(
                        f"refusing to merge: meta['{k}'] differs between "
                        f"{args.inputs[0]} and {name}")

        for s in SECTIONS:
            for r in d.get(s, []):
                k = key(r)
                if k in seen[s]:
                    raise SystemExit(f"duplicate {s} run {k} -- would double-count")
                seen[s].add(k)
                merged[s].append(r)

        for a in d["meta"].get("archs", []):
            if a not in archs:
                archs.append(a)

    merged["meta"]["archs"] = archs
    merged["meta"]["merged_from"] = list(args.inputs)
    merged["meta"]["total_seconds"] = None

    out = os.path.join(RESULTS_DIR, args.out)
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)

    print(f"Merged {len(args.inputs)} files -> {out}")
    print(f"  architectures: {archs}")
    for s in SECTIONS:
        print(f"  {s}: {len(merged[s])} runs")


if __name__ == "__main__":
    main()
