from pathlib import Path

import numpy as np


COLORS = {"h264": "tab:blue", "h265": "tab:orange", "bistar": "tab:green", "bistar_atlas": "tab:red"}


def plot_roi_preview(sources, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for (phase, sequence), source in sources.items():
        indices = sorted({0, len(source["frames"]) // 2, len(source["frames"]) - 1})
        fig, axes = plt.subplots(1, len(indices) + 1, figsize=(4 * (len(indices) + 1), 4))
        for axis, index in zip(axes[:-1], indices):
            axis.imshow(source["frames"][index])
            overlay = np.zeros((*source["masks"][index].shape, 4))
            overlay[source["masks"][index]] = [0, 1, 0.3, 0.3]
            axis.imshow(overlay)
            axis.set_title("Annotated ROI | frame {}".format(source["frame_ids"][index]))
            axis.axis("off")
        frequency = np.mean(source["masks"], axis=0)
        image = axes[-1].imshow(frequency, cmap="magma", vmin=0, vmax=1)
        axes[-1].set_title("Independent ROI frequency")
        fig.colorbar(image, ax=axes[-1], shrink=0.7)
        fig.suptitle("{} | {} | annotation audit".format(sequence, phase))
        fig.tight_layout()
        fig.savefig(output_dir / "{}_{}_ROI_preview.png".format(sequence, phase), dpi=160)
        plt.close(fig)


def plot_search_results(calibration, validation, comparisons, patch_rows, output_dir, catalog=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.reports.plots import plot_patch_heatmaps
    from src.experiments.config_search import OBJECTIVES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {candidate["candidate_id"]: ", ".join(
        "{}={}".format(key.split(".")[-1], value) for key, value in candidate.get("overrides", {}).items()
    ) for candidate in (catalog or [])}
    for method in COLORS:
        current = [row for row in calibration if row["method"] == method]
        candidate_ids = sorted({row["candidate_id"] for row in current})
        if not candidate_ids:
            continue
        rates = sorted({row["target_kbps"] for row in current})
        values = np.full((len(candidate_ids), len(rates)), np.nan)
        for i, candidate_id in enumerate(candidate_ids):
            for j, rate in enumerate(rates):
                samples = [row["roi_ssim"] for row in current if row["candidate_id"] == candidate_id
                           and row["target_kbps"] == rate and row.get("roi_ssim") is not None]
                if samples:
                    values[i, j] = np.mean(samples)
        fig, axis = plt.subplots(figsize=(13, max(4, len(candidate_ids) * 0.35)))
        image = axis.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(rates)), [str(rate) for rate in rates])
        axis.set_yticks(range(len(candidate_ids)), ["{} | {}".format(candidate_id[-6:], labels.get(candidate_id, ""))
                                                   for candidate_id in candidate_ids], fontsize=7)
        axis.set(xlabel="Target bitrate (kbps)", ylabel="Configuration (see candidate_catalog.csv)",
                 title="{} calibration ROI SSIM; blank = unmatched or failed".format(method))
        fig.colorbar(image, ax=axis, label="Mean ROI SSIM across sequences")
        fig.tight_layout()
        fig.savefig(output_dir / (method + "_configuration_heatmap.png"), dpi=180)
        plt.close(fig)
    for metric, label in (("roi_ssim", "Critical ROI SSIM (higher is better)"),
                          ("roi_psnr", "Critical ROI PSNR (dB)"),
                          ("critical_patch_retention", "Critical-patch retention rate")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for method, color in COLORS.items():
            matched = [row for row in calibration if row["method"] == method and row.get(metric) is not None]
            axes[0].scatter([row["actual_bitrate_kbps"] for row in matched], [row[metric] for row in matched],
                            color=color, alpha=0.4, s=18, label=method)
            valid = [row for row in validation if row["method"] == method and row.get(metric) is not None]
            rates = sorted({row["target_kbps"] for row in valid})
            values = [np.mean([row[metric] for row in valid if row["target_kbps"] == rate]) for rate in rates]
            actual = [np.mean([row["actual_bitrate_kbps"] for row in valid if row["target_kbps"] == rate]) for rate in rates]
            axes[1].plot(actual, values, "o-", color=color, label=method)
        axes[0].set_title("Calibration: all matched configurations")
        axes[1].set_title("Held-out: frozen selected configurations")
        for axis in axes:
            axis.set(xlabel="Actual bitrate (kbps)", ylabel=label)
            axis.grid(alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / (metric + "_rate_comparison.png"), dpi=180)
        fig.savefig(output_dir / (metric + "_rate_comparison.pdf"))
        plt.close(fig)

    for method, baseline in (("bistar", "h264"), ("bistar", "h265"),
                             ("bistar_atlas", "h264"), ("bistar_atlas", "h265")):
        current = [row for row in comparisons if row["method"] == method and row["baseline"] == baseline]
        if not current:
            continue
        rates = sorted({row["target_kbps"] for row in current})
        means, errors, labels = [], [], []
        for rate in rates:
            samples = [float(row["delta"]) for row in current if row["target_kbps"] == rate and row["comparable"]]
            means.append(float(np.mean(samples)) if samples else 0.0)
            errors.append(float(np.std(samples, ddof=1) / np.sqrt(len(samples))) if len(samples) > 1 else 0.0)
            labels.append("{}\nn={}".format(rate, len(samples)) if samples else "{}\nunmatched".format(rate))
        fig, axis = plt.subplots(figsize=(7, 4))
        objective = current[0]["objective"]
        axis.bar(range(len(rates)), means, yerr=errors,
                 color=["tab:green" if (value > 0 if OBJECTIVES[objective] else value < 0)
                        else "tab:red" for value in means], capsize=3)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(range(len(rates)), labels)
        axis.set(xlabel="Target bitrate (kbps)", ylabel="{} difference".format(objective),
                 title="{} - {} (n = independent sequences; bars: mean +/- SE)".format(method, baseline))
        fig.tight_layout()
        fig.savefig(output_dir / "{}_vs_{}_gain.png".format(method, baseline), dpi=180)
        plt.close(fig)

    if patch_rows:
        typed = [{**row, "target_kbps": float(row["target_kbps"])} for row in patch_rows]
        plot_patch_heatmaps(typed, output_dir / "spatial")
        for sequence, target in sorted({(row["sequence_id"], row["target_kbps"]) for row in typed}):
            current = [row for row in typed if row["sequence_id"] == sequence and row["target_kbps"] == target]
            by_method = {method: {int(row["patch_index"]): row for row in current if row["method"] == method}
                         for method in COLORS}
            for baseline in ("h264", "h265"):
                if not by_method[baseline]:
                    continue
                differences = []
                names = []
                for method in ("bistar", "bistar_atlas"):
                    if not by_method[method]:
                        continue
                    height = max(int(row["patch_row"]) for row in by_method[method].values()) + 1
                    width = max(int(row["patch_column"]) for row in by_method[method].values()) + 1
                    values = np.zeros((height, width))
                    for index, row in by_method[method].items():
                        values[int(row["patch_row"]), int(row["patch_column"])] = (
                            float(row["patch_ssim"]) - float(by_method[baseline][index]["patch_ssim"]))
                    differences.append(values)
                    names.append(method)
                if not differences:
                    continue
                limit = max(1e-6, max(float(np.abs(array).max()) for array in differences))
                fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
                for axis, name, values in zip(axes[0], names, differences):
                    image = axis.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit)
                    axis.set(title="{} - {}".format(name, baseline), xlabel="Patch column", ylabel="Patch row")
                fig.colorbar(image, ax=axes.ravel().tolist(), label="Patch SSIM gain", shrink=0.8)
                fig.suptitle("{} | {} kbps | held-out spatial difference".format(sequence, target))
                fig.subplots_adjust(top=0.8, wspace=0.3)
                fig.savefig(output_dir / "spatial" / "{}_{}kbps_gain_vs_{}.png".format(sequence, target, baseline), dpi=180)
                plt.close(fig)
