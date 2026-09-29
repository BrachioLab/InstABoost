#!/usr/bin/env python
"""Paired statistics comparing IFEval methods on the same model.

Reads results/ifeval/{model_slug}/{method}/results_by_sample.json (same
400 prompts, keyed identically across methods for a given model) and, for
every pair of methods available for that model, computes:

  Sample-level (prompt-strict accuracy):
    - accuracy diff (A - B) with a paired bootstrap 95% CI (resampling
      sample indices jointly across methods, not independently per method)
    - exact two-sided McNemar test on the discordant pairs

  Instruction-level (per-sample mean instruction score as the paired unit):
    - accuracy diff (A - B) with the same paired bootstrap CI
    - Wilcoxon signed-rank test (normal approximation, tie-corrected) on
      the paired per-sample differences

p-values are Holm-Bonferroni adjusted within each (model, level) family of
comparisons. Uses only the standard library (no numpy/scipy dependency) so
it runs with any Python >= 3.8 that has results_by_sample.json on disk.

Usage:
    python src/ifeval_paired_stats.py
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "ifeval"
OUT_MD = REPO_ROOT / "results" / "ifeval_paired_stats.md"
OUT_JSON = REPO_ROOT / "results" / "ifeval_paired_stats.json"

MODEL_DISPLAY = {
    "Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "gemma-7b-it": "Gemma-7b-it",
    "Qwen3-14B": "Qwen3-14B",
    "gpt-oss-20b": "GPT-OSS-20B",
}
METHOD_ORDER = ["prompt", "pasta", "spotlight", "instaboost"]

N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 12345
ALPHA = 0.05


def load_samples(model_slug: str, method: str) -> dict | None:
    path = RESULTS_ROOT / model_slug / method / "results_by_sample.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def discover_methods(model_slug: str) -> list[str]:
    present = []
    for method in METHOD_ORDER:
        if (RESULTS_ROOT / model_slug / method / "results_by_sample.json").exists():
            present.append(method)
    return present


def aligned_vectors(samples_a: dict, samples_b: dict, field: str):
    keys = sorted(set(samples_a) & set(samples_b), key=int)
    a = [samples_a[k][field] for k in keys]
    b = [samples_b[k][field] for k in keys]
    return a, b, keys


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def binom_two_sided_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value via the small-p-value method."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    observed = pmf[k]
    tol = observed * (1 + 1e-9)
    total = sum(x for x in pmf if x <= tol)
    return min(total, 1.0)


def mcnemar_exact(a_scores: list[int], b_scores: list[int]) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes."""
    b = sum(1 for x, y in zip(a_scores, b_scores) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(a_scores, b_scores) if x == 0 and y == 1)
    n = b + c
    k = min(b, c)
    p = binom_two_sided_pvalue(k, n, 0.5)
    return {"b_a_only": b, "c_b_only": c, "n_discordant": n, "p_value": p}


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test, normal approximation with tie
    correction and continuity correction. Zero diffs are dropped."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return {"n_nonzero": 0, "statistic": None, "p_value": 1.0}

    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    tie_correction = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        group_size = j - i + 1
        tie_correction += group_size ** 3 - group_size
        for idx in order[i : j + 1]:
            ranks[idx] = avg_rank
        i = j + 1

    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    stat = min(w_pos, w_neg)
    mean_t = n * (n + 1) / 4.0
    var_t = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    if var_t <= 0:
        return {"n_nonzero": n, "statistic": stat, "p_value": 1.0}
    if stat < mean_t:
        z = (stat - mean_t + 0.5) / math.sqrt(var_t)
    else:
        z = (stat - mean_t - 0.5) / math.sqrt(var_t)
    p = 2 * (1 - norm_cdf(abs(z)))
    return {"n_nonzero": n, "statistic": stat, "p_value": min(p, 1.0)}


def paired_bootstrap_diff_ci(
    a: list[float],
    b: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Resamples sample indices jointly (paired), returns (point, lo, hi)
    for the diff of means A - B."""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    idx_range = range(n)
    boot_means = []
    for _ in range(n_bootstrap):
        idxs = rng.choices(idx_range, k=n)
        boot_means.append(sum(diffs[i] for i in idxs) / n)
    boot_means.sort()
    lo_idx = int((1 - confidence) / 2 * n_bootstrap)
    hi_idx = min(int((1 + confidence) / 2 * n_bootstrap), n_bootstrap - 1)
    point = sum(diffs) / n
    return point, boot_means[lo_idx], boot_means[hi_idx]


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, returns adjusted p-values in
    the same order as the input."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def analyze_model(model_slug: str) -> dict | None:
    methods = discover_methods(model_slug)
    if len(methods) < 2:
        return None
    samples = {m: load_samples(model_slug, m) for m in methods}
    pairs = list(itertools.combinations(methods, 2))

    sample_level = []
    instruction_level = []
    for method_a, method_b in pairs:
        a_scores, b_scores, keys = aligned_vectors(
            samples[method_a], samples[method_b], "sample_score"
        )
        acc_a = sum(a_scores) / len(a_scores)
        acc_b = sum(b_scores) / len(b_scores)
        diff, lo, hi = paired_bootstrap_diff_ci(a_scores, b_scores)
        mcnemar = mcnemar_exact(a_scores, b_scores)
        sample_level.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n": len(keys),
                "acc_a": acc_a,
                "acc_b": acc_b,
                "diff": diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": mcnemar["p_value"],
                "b": mcnemar["b_a_only"],
                "c": mcnemar["c_b_only"],
            }
        )

        ai, bi, _ = aligned_vectors(
            samples[method_a], samples[method_b], "instruction_level_score"
        )
        acc_ai = sum(ai) / len(ai)
        acc_bi = sum(bi) / len(bi)
        diff_i, lo_i, hi_i = paired_bootstrap_diff_ci(ai, bi)
        diffs_i = [x - y for x, y in zip(ai, bi)]
        wilcoxon = wilcoxon_signed_rank(diffs_i)
        instruction_level.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n": len(ai),
                "acc_a": acc_ai,
                "acc_b": acc_bi,
                "diff": diff_i,
                "ci_lo": lo_i,
                "ci_hi": hi_i,
                "p_value": wilcoxon["p_value"],
                "n_nonzero": wilcoxon["n_nonzero"],
            }
        )

    sample_p_adj = holm_adjust([r["p_value"] for r in sample_level])
    for r, p_adj in zip(sample_level, sample_p_adj):
        r["p_value_holm"] = p_adj
    instr_p_adj = holm_adjust([r["p_value"] for r in instruction_level])
    for r, p_adj in zip(instruction_level, instr_p_adj):
        r["p_value_holm"] = p_adj

    result = {"methods": methods, "sample_level": sample_level, "instruction_level": instruction_level}
    if "instaboost" in methods:
        result["instaboost_vs_others_sample"] = instaboost_focus(sample_level)
        result["instaboost_vs_others_instruction"] = instaboost_focus(instruction_level)
    return result


def instaboost_focus(rows: list[dict]) -> list[dict]:
    """Extracts the `X vs instaboost` rows and re-runs Holm-Bonferroni
    within just that family (instead of the full pairwise grid), since
    that's the family of hypotheses actually being asked about. Flips
    sign so `diff`/`ci_lo`/`ci_hi` read as instaboost - other."""
    focused = [r for r in rows if r["method_b"] == "instaboost"]
    p_adj = holm_adjust([r["p_value"] for r in focused])
    out = []
    for r, p in zip(focused, p_adj):
        diff = -r["diff"]
        lo, hi = -r["ci_hi"], -r["ci_lo"]
        if p >= ALPHA:
            verdict = "no significant difference"
        elif diff > 0:
            verdict = "instaboost significantly better"
        else:
            verdict = "instaboost significantly worse"
        out.append(
            {
                "other_method": r["method_a"],
                "n": r["n"],
                "acc_instaboost": r["acc_b"],
                "acc_other": r["acc_a"],
                "diff": diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_value": r["p_value"],
                "p_value_holm": p,
                "verdict": verdict,
            }
        )
    return out


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}"


def fmt_diff(x: float) -> str:
    formatted = f"{x * 100:.1f}"
    if formatted == "-0.0":
        formatted = "0.0"
    sign = "" if formatted.startswith("-") else "+"
    return f"{sign}{formatted}"


def sig_marker(p_holm: float) -> str:
    if p_holm < 0.001:
        return "***"
    if p_holm < 0.01:
        return "**"
    if p_holm < ALPHA:
        return "*"
    return ""


def verdict_symbol(entry: dict | None) -> str:
    if entry is None:
        return "n/a"
    if entry["verdict"] == "instaboost significantly better":
        return "✓"
    if entry["verdict"] == "instaboost significantly worse":
        return "✗"
    return "–"


def write_markdown(results: dict) -> None:
    lines: list[str] = []
    lines.append("# IFEval paired statistics")
    lines.append("")
    lines.append(
        "Paired comparisons between methods evaluated on the **same 400 prompts** "
        "for a given model (matched by sample key). Each comparison reports:"
    )
    lines.append("")
    lines.append(
        "- **Sample-level (prompt-strict)**: accuracy diff (A - B) with a paired "
        "bootstrap 95% CI (resampling sample indices jointly across methods, "
        f"{N_BOOTSTRAP} resamples), and an exact two-sided **McNemar test** p-value "
        "on the discordant pairs (b = A-only-correct, c = B-only-correct)."
    )
    lines.append(
        "- **Instruction-level**: accuracy diff (A - B) using each sample's mean "
        "per-instruction score as the paired unit, with the same paired bootstrap "
        "CI, and a **Wilcoxon signed-rank test** p-value (normal approximation, "
        "tie-corrected) on the paired per-sample differences."
    )
    lines.append("")
    lines.append(
        f"p-values are Holm-Bonferroni adjusted within each (model, level) family "
        f"of comparisons. `*` p<{ALPHA}, `**` p<0.01, `***` p<0.001 (adjusted)."
    )
    lines.append("")

    lines.append("## Instaboost vs. other methods — significance at a glance")
    lines.append("")
    lines.append(
        "Same instaboost-vs-X family and Holm correction as the detailed table below, "
        "with the numbers stripped out. `✓` = instaboost significantly better, "
        "`✗` = instaboost significantly worse, `–` = no significant difference, "
        "`n/a` = that method wasn't run for this model."
    )
    lines.append("")
    other_methods = [m for m in METHOD_ORDER if m != "instaboost"]
    header_cells = ["Model"]
    for m in other_methods:
        header_cells.append(f"{m} (sample)")
        header_cells.append(f"{m} (instr)")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|---|" + ":---:|" * (2 * len(other_methods)))
    for model_slug, display in MODEL_DISPLAY.items():
        if model_slug not in results:
            continue
        res = results[model_slug]
        sample_by_method = {
            r["other_method"]: r for r in res.get("instaboost_vs_others_sample", [])
        }
        instr_by_method = {
            r["other_method"]: r for r in res.get("instaboost_vs_others_instruction", [])
        }
        row = [display]
        for m in other_methods:
            row.append(verdict_symbol(sample_by_method.get(m)))
            row.append(verdict_symbol(instr_by_method.get(m)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Instaboost vs. other methods — summary")
    lines.append("")
    lines.append(
        "Same paired data as above, but the Holm-Bonferroni correction is applied "
        "only within the *instaboost-vs-X* family (2-3 tests per model per level, "
        "not all 6 pairwise combinations), since that's the actual family of "
        "hypotheses being asked about here. Diff is **instaboost - other** (positive "
        "= instaboost better)."
    )
    lines.append("")
    for level, key in (("Sample-level", "instaboost_vs_others_sample"), ("Instruction-level", "instaboost_vs_others_instruction")):
        lines.append(f"### {level}")
        lines.append("")
        lines.append("| Model | vs. | n | acc(instaboost) | acc(other) | diff pp | 95% CI (pp) | p (Holm) | verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---|---:|---|")
        for model_slug, display in MODEL_DISPLAY.items():
            if model_slug not in results or key not in results[model_slug]:
                continue
            for r in results[model_slug][key]:
                lines.append(
                    f"| {display} | {r['other_method']} | {r['n']} | "
                    f"{fmt_pct(r['acc_instaboost'])} | {fmt_pct(r['acc_other'])} | "
                    f"{fmt_diff(r['diff'])} | [{fmt_diff(r['ci_lo'])}, {fmt_diff(r['ci_hi'])}] | "
                    f"{r['p_value_holm']:.4f}{sig_marker(r['p_value_holm'])} | {r['verdict']} |"
                )
        lines.append("")

    for model_slug, display in MODEL_DISPLAY.items():
        if model_slug not in results:
            continue
        res = results[model_slug]
        lines.append(f"## {display} — full pairwise grid (appendix)")
        lines.append("")
        lines.append(f"Methods available: {', '.join(res['methods'])}")
        lines.append("")

        lines.append("### Sample-level (prompt-strict accuracy)")
        lines.append("")
        lines.append(
            "| A vs B | n | acc(A) | acc(B) | diff (A-B) pp | 95% CI (pp) | "
            "McNemar p (raw) | p (Holm) | |"
        )
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|---|")
        for r in res["sample_level"]:
            lines.append(
                f"| {r['method_a']} vs {r['method_b']} | {r['n']} | "
                f"{fmt_pct(r['acc_a'])} | {fmt_pct(r['acc_b'])} | {fmt_diff(r['diff'])} | "
                f"[{fmt_diff(r['ci_lo'])}, {fmt_diff(r['ci_hi'])}] | "
                f"{r['p_value']:.4f} | {r['p_value_holm']:.4f} | "
                f"{sig_marker(r['p_value_holm'])} |"
            )
        lines.append("")

        lines.append("### Instruction-level (per-sample mean instruction score)")
        lines.append("")
        lines.append(
            "| A vs B | n | acc(A) | acc(B) | diff (A-B) pp | 95% CI (pp) | "
            "Wilcoxon p (raw) | p (Holm) | |"
        )
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|---|")
        for r in res["instruction_level"]:
            lines.append(
                f"| {r['method_a']} vs {r['method_b']} | {r['n']} | "
                f"{fmt_pct(r['acc_a'])} | {fmt_pct(r['acc_b'])} | {fmt_diff(r['diff'])} | "
                f"[{fmt_diff(r['ci_lo'])}, {fmt_diff(r['ci_hi'])}] | "
                f"{r['p_value']:.4f} | {r['p_value_holm']:.4f} | "
                f"{sig_marker(r['p_value_holm'])} |"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


def main() -> None:
    results = {}
    for model_slug in MODEL_DISPLAY:
        analyzed = analyze_model(model_slug)
        if analyzed is not None:
            results[model_slug] = analyzed
    if not results:
        raise SystemExit(f"No comparable results found under {RESULTS_ROOT}")
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}")
    write_markdown(results)


if __name__ == "__main__":
    main()
