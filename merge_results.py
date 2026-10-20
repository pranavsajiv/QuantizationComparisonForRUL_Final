"""
Merge several run_study.py output files into one results.json.

run_study.py can be invoked repeatedly with different --seeds so that extra
seeds can be added without discarding compute already spent. This script
concatenates the run records and unions the seed lists, refusing to merge
files whose experimental setup does not match.

    python merge_results.py results_a.json results_b.json --out results.json
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Fields that must agree across files for a merge to be meaningful
CRITICAL = ["subset", "seq_len", "rul_cap", "features", "patience",
            "batch_size", "learning_rate", "calibration_samples",
            "quantization", "thresholds", "y_true", "tensorflow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    merged, seeds, archs, seen = None, [], [], set()

    for name in args.inputs:
        path = name if os.path.isabs(name) else os.path.join(RESULTS_DIR, name)
        with open(path) as fh:
            d = json.load(fh)

        if merged is None:
            merged = d
        else:
            for k in CRITICAL:
                if merged["meta"].get(k) != d["meta"].get(k):
                    raise SystemExit(
                        f"refusing to merge: meta['{k}'] differs between "
                        f"{args.inputs[0]} and {name}")
            if merged["meta"].get("smoke") or d["meta"].get("smoke"):
                raise SystemExit("refusing to merge a --smoke run into real results")
            merged["runs"].extend(d["runs"])

        for s in d["meta"]["seeds"]:
            if s not in seeds:
                seeds.append(s)
        for a in d["meta"]["archs"]:
            if a not in archs:
                archs.append(a)

    for r in merged["runs"]:
        key = (r["arch"], r["seed"])
        if key in seen:
            raise SystemExit(f"duplicate run for {key} -- would double-count")
        seen.add(key)

    merged["meta"]["seeds"] = seeds
    merged["meta"]["archs"] = archs
    merged["meta"]["merged_from"] = list(args.inputs)
    merged["meta"]["total_seconds"] = None  # no longer a single wall-clock

    out = os.path.join(RESULTS_DIR, args.out)
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)

    print(f"Merged {len(args.inputs)} files -> {out}")
    print(f"  {len(merged['runs'])} runs, {len(archs)} architectures, "
          f"{len(seeds)} seeds: {seeds}")


if __name__ == "__main__":
    main()
