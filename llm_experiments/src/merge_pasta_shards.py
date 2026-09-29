"""Merge PASTA per-(layer, head) shard JSONs into a single file.

Walks ``{pasta_dir}/layer_head_selection_results.shard_*.json``, dict-updates
all (layer, head) entries into one dict, writes
``{pasta_dir}/layer_head_selection_results.json``. Overlapping entries between
shards are deduped — last-shard-wins (scores are deterministic at fixed alpha,
so duplicates should match within bf16 noise; if they don't, that's a signal
the shards weren't run with identical config).

Usage:
    python src/merge_pasta_shards.py results_pasta/ifeval/Qwen3-14B/pasta
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pasta_dir",
        help="Directory containing layer_head_selection_results.shard_*.json files",
    )
    ap.add_argument(
        "--output_name",
        default="layer_head_selection_results.json",
        help="Output filename within pasta_dir (default matches the unsharded path).",
    )
    args = ap.parse_args()

    shards = sorted(
        glob.glob(os.path.join(args.pasta_dir, "layer_head_selection_results.shard_*.json"))
    )
    if not shards:
        raise SystemExit(f"No shard files in {args.pasta_dir}")

    combined: dict[str, dict[str, dict]] = defaultdict(dict)
    overlap_count = 0
    for shard in shards:
        n_before = sum(len(v) for v in combined.values())
        with open(shard) as f:
            d = json.load(f)
        for layer, heads in d.items():
            for head, entry in heads.items():
                if head in combined[layer]:
                    overlap_count += 1
                combined[layer][head] = entry
        n_after = sum(len(v) for v in combined.values())
        print(f"  + {os.path.basename(shard):60s}  added {n_after - n_before:>4} new")

    out_path = os.path.join(args.pasta_dir, args.output_name)
    sorted_combined = {
        str(layer): {
            str(head): combined[str(layer)][str(head)]
            for head in sorted(combined[str(layer)].keys(), key=int)
        }
        for layer in sorted((int(k) for k in combined.keys()))
    }
    with open(out_path, "w") as f:
        json.dump(sorted_combined, f, indent=2)

    n_layers = len(sorted_combined)
    n_entries = sum(len(v) for v in sorted_combined.values())
    print(f"\nWrote {out_path}")
    print(f"  layers: {n_layers}  entries: {n_entries}  overlap (deduped): {overlap_count}")


if __name__ == "__main__":
    main()
