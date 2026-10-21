"""Post-process an ROI search without using held-out data to select configurations."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.config_search import OBJECTIVES


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _mean_field(rows, field):
    values = [_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else ""


def _bootstrap_interval(values, generator, draws=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = generator.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def _dataset(sequence_id):
    return sequence_id.split("_", 1)[0]


def analyze_search_output(output_dir, bootstrap_draws=2000, seed=0):
    """Write calibration ranks and held-out paired summaries to a completed run."""
    output = Path(output_dir)
    calibration = _read_csv(output / "calibration_results.csv")
    validation = _read_csv(output / "validation_results.csv")
    comparisons = _read_csv(output / "comparisons.csv")
    catalog = {row["candidate_id"]: row for row in _read_csv(output / "candidate_catalog.csv")}
    winners = _read_csv(output / "best_configs.csv")
    snapshot = (output / "config_snapshot.yaml")
    if not snapshot.is_file():
        raise ValueError("Missing immutable run snapshot: {}".format(snapshot))
    run_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    objective = run_summary["objective"]
    high_is_good = OBJECTIVES[objective]
    rng = np.random.default_rng(seed)

    by_candidate = defaultdict(list)
    for row in calibration:
        by_candidate[(row["method"], row["target_kbps"], row["candidate_id"])].append(row)
    rankings = []
    for (method, rate, candidate_id), rows in by_candidate.items():
        matched = [row for row in rows if row["status"] == "matched" and _number(row.get(objective)) is not None]
        scores = [_number(row[objective]) for row in matched]
        rankings.append({"method": method, "target_kbps": rate, "candidate_id": candidate_id,
                         "overrides": catalog.get(candidate_id, {}).get("overrides", "{}"),
                         "matched_sequences": len(matched), "total_sequences": len(rows),
                         "mean_calibration_score": float(np.mean(scores)) if scores else "",
                         "mean_actual_kbps": _mean_field(matched, "actual_bitrate_kbps"),
                         "eligible_for_selection": len(matched) == len(rows)})
    rankings.sort(key=lambda row: (row["method"], float(row["target_kbps"]),
                                    not row["eligible_for_selection"],
                                    -row["mean_calibration_score"] if high_is_good and row["mean_calibration_score"] != "" else
                                    row["mean_calibration_score"] if row["mean_calibration_score"] != "" else float("inf")))
    _write_csv(output / "formal_candidate_ranking.csv", rankings,
               ["method", "target_kbps", "candidate_id", "overrides", "matched_sequences",
                "total_sequences", "mean_calibration_score", "mean_actual_kbps", "eligible_for_selection"])

    by_pair = defaultdict(list)
    for row in comparisons:
        by_pair[(row["method"], row["baseline"], row["target_kbps"], "all")].append(row)
        by_pair[(row["method"], row["baseline"], row["target_kbps"], _dataset(row["sequence_id"]))].append(row)
    paired = []
    for (method, baseline, rate, dataset), rows in sorted(by_pair.items()):
        comparable = [row for row in rows if row["comparable"].lower() == "true" and _number(row["delta"]) is not None]
        deltas = [_number(row["delta"]) for row in comparable]
        oriented = [value if high_is_good else -value for value in deltas]
        low, high = _bootstrap_interval(deltas, rng, bootstrap_draws)
        paired.append({"method": method, "baseline": baseline, "target_kbps": rate,
                       "dataset": dataset, "n_comparable": len(comparable), "n_total": len(rows),
                       "mean_delta": float(np.mean(deltas)) if deltas else "",
                       "ci95_low": low if low is not None else "", "ci95_high": high if high is not None else "",
                       "win_count": sum(value > 0 for value in oriented),
                       "win_fraction": float(np.mean(np.asarray(oriented) > 0)) if oriented else "",
                       "mean_pair_rate_gap_percent": float(np.mean([_number(row["pair_rate_gap_percent"])
                                                                     for row in comparable])) if comparable else ""})
    _write_csv(output / "formal_paired_summary.csv", paired,
               ["method", "baseline", "target_kbps", "dataset", "n_comparable", "n_total",
                "mean_delta", "ci95_low", "ci95_high", "win_count", "win_fraction", "mean_pair_rate_gap_percent"])

    by_validation = defaultdict(list)
    for row in validation:
        by_validation[(row["method"], row["target_kbps"])].append(row)
    method_rows = []
    for (method, rate), rows in sorted(by_validation.items()):
        matched = [row for row in rows if row["status"] == "matched" and _number(row.get(objective)) is not None]
        method_rows.append({"method": method, "target_kbps": rate, "n_matched": len(matched),
                            "n_total": len(rows),
                            "mean_actual_kbps": _mean_field(matched, "actual_bitrate_kbps"),
                            "mean_objective": _mean_field(matched, objective),
                            "mean_roi_psnr": _mean_field(matched, "roi_psnr"),
                            "mean_latency_ms": _mean_field(matched, "mean_latency_ms")})
    _write_csv(output / "formal_method_summary.csv", method_rows,
               ["method", "target_kbps", "n_matched", "n_total", "mean_actual_kbps",
                "mean_objective", "mean_roi_psnr", "mean_latency_ms"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, method in zip(axes, ("bistar", "bistar_atlas")):
        current = [row for row in paired if row["method"] == method and row["dataset"] == "all"]
        rates = sorted({float(row["target_kbps"]) for row in current})
        for baseline, color, offset in (("h264", "tab:blue", -0.12), ("h265", "tab:orange", 0.12)):
            data = {float(row["target_kbps"]): row for row in current if row["baseline"] == baseline}
            for index, rate in enumerate(rates):
                row = data.get(rate)
                if row is None or row["mean_delta"] == "":
                    continue
                x = index + offset
                mean = row["mean_delta"]
                axis.errorbar(x, mean, yerr=[[mean - row["ci95_low"]], [row["ci95_high"] - mean]],
                              fmt="o", color=color, capsize=3, label=baseline if index == 0 else None)
                axis.annotate("{}/{}".format(row["win_count"], row["n_comparable"]),
                              (x, mean), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(rates)), ["{:g}".format(rate) for rate in rates])
        axis.set(xlabel="Target bitrate (kbps)", title=method)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Baseline")
    axes[0].set_ylabel("Paired {} difference (method - baseline)".format(objective))
    fig.suptitle("Held-out matched-rate ROI fidelity; 95% sequence bootstrap CI; labels = wins / matched pairs")
    fig.tight_layout()
    fig.savefig(plots / "formal_paired_roi_difference.png", dpi=180)
    fig.savefig(plots / "formal_paired_roi_difference.pdf")
    plt.close(fig)

    selected = {(row["method"], row["target_kbps"]): row for row in winners}
    positive_pairs = sum(row["win_count"] for row in paired if row["dataset"] == "all")
    comparable_pairs = sum(row["n_comparable"] for row in paired if row["dataset"] == "all")
    datasets = sorted({_dataset(row["sequence_id"]) for row in validation})
    lines = ["# Formal ROI search analysis", "",
             "The selected configuration for each method and target bitrate is frozen using calibration videos. The held-out videos are used only for evaluation.",
             "", "Objective: `{}`; {} method-minus-baseline difference favors the method.".format(
                 objective, "positive" if high_is_good else "negative"),
             "Held-out coverage: {} sequences from {}. Comparable pair wins: {} / {}. {}".format(
                 len({row["sequence_id"] for row in validation}), ", ".join(datasets),
                 positive_pairs, comparable_pairs,
                 "No measured pair supports superiority." if positive_pairs == 0 else
                 "Inspect each target, baseline and confidence interval before making a superiority claim."),
             "Sequence bootstrap: {} draws, seed {}. These intervals describe the observed video sample; they do not establish teleoperation task success.".format(bootstrap_draws, seed),
             "", "## Selected configurations", "",
             "| Target kbps | Method | Candidate | Calibration score | Overrides |", "| ---: | --- | --- | ---: | --- |"]
    for (method, rate), row in sorted(selected.items(), key=lambda item: (float(item[0][1]), item[0][0])):
        overrides = catalog.get(row["candidate_id"], {}).get("overrides", "{}")
        lines.append("| {} | {} | `{}` | {:.4f} | `{}` |".format(rate, method, row["candidate_id"],
                     float(row["calibration_score"]), overrides))
    lines += ["", "## Held-out matched-rate differences", "",
              "Only same-video pairs with both methods matched and an actual rate gap inside the configured tolerance contribute to the differences below.",
              "", "| Method | Baseline | Target kbps | Comparable / total | Mean difference | 95% bootstrap CI | Wins |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in paired:
        if row["dataset"] != "all":
            continue
        mean = "{:+.4f}".format(row["mean_delta"]) if row["mean_delta"] != "" else "n/a"
        ci = "[{:+.4f}, {:+.4f}]".format(row["ci95_low"], row["ci95_high"]) if row["mean_delta"] != "" else "n/a"
        lines.append("| {} | {} | {} | {} / {} | {} | {} | {} |".format(
            row["method"], row["baseline"], row["target_kbps"], row["n_comparable"],
            row["n_total"], mean, ci, row["win_count"]))
    lines += ["", "![Held-out paired ROI difference](plots/formal_paired_roi_difference.png)", "",
              "Per-dataset breakdown: [formal_paired_summary.csv](formal_paired_summary.csv). All calibration candidates: [formal_candidate_ranking.csv](formal_candidate_ranking.csv). Method rates and latency: [formal_method_summary.csv](formal_method_summary.csv).",
              "", "The ROI measurements use the run's task/gaze annotations (provenance: {}). The result concerns visual fidelity in annotated regions, not closed-loop remote-control performance. The measured rate includes codec/application bytes and excludes network headers. Missing or rate-unmatched pairs cannot support superiority claims. Bootstrap intervals are exploratory and unadjusted for multiple comparisons.".format(
                  ", ".join(run_summary.get("annotation_provenance", []))),
              "", "Reproduction metadata and raw observations are in `config_snapshot.yaml`, `source_metadata.json`, `candidate_catalog.csv`, `calibration_results.csv`, `validation_results.csv`, and `comparisons.csv`."]
    (output / "formal_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {"objective": objective, "bootstrap_draws": bootstrap_draws, "seed": seed,
                "calibration_candidates": len(rankings), "selected_configs": len(winners),
                "held_out_sequences": len({row["sequence_id"] for row in validation}),
                "comparable_pairs": sum(row["n_comparable"] for row in paired if row["dataset"] == "all"),
                "interpretation": "positive method-minus-baseline objective differences favor the method" if high_is_good else
                                  "negative method-minus-baseline objective differences favor the method"}
    (output / "formal_analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
