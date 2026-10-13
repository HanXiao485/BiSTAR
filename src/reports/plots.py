from pathlib import Path


def plot_rd(rows, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("psnr", "ssim", "lpips"):
        fig, axis = plt.subplots()
        for method in sorted(set(row["method"] for row in rows)):
            points = sorted((row for row in rows if row["method"] == method), key=lambda x: x["actual_bitrate_kbps"])
            axis.plot([p["actual_bitrate_kbps"] for p in points], [p[metric] for p in points], marker="o", label=method)
        axis.set(xlabel="Actual bitrate (kbps)", ylabel=metric.upper()); axis.grid(True); axis.legend(); fig.tight_layout()
        fig.savefig(output_dir / "bitrate_vs_{}.png".format(metric)); plt.close(fig)


def plot_patch_heatmaps(rows, output_dir: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("patch_psnr", "Patch PSNR (dB)", "viridis"),
        ("patch_ssim", "Patch SSIM", "viridis"),
        ("patch_normalized_l1", "Patch normalized L1", "viridis_r"),
        ("selection_frequency", "Selection frequency", "magma"),
    )
    groups = sorted({(row["sequence_id"], row["target_kbps"]) for row in rows})
    for sequence_id, target in groups:
        selected = [row for row in rows if row["sequence_id"] == sequence_id and row["target_kbps"] == target]
        methods = sorted({row["method"] for row in selected})
        for metric, label, color_map in metrics:
            arrays = []
            for method in methods:
                current = [row for row in selected if row["method"] == method]
                height = max(int(row["patch_row"]) for row in current) + 1
                width = max(int(row["patch_column"]) for row in current) + 1
                values = np.full((height, width), np.nan)
                for row in current:
                    values[int(row["patch_row"]), int(row["patch_column"])] = float(row[metric])
                arrays.append(values)
            finite_parts = [array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()]
            finite = np.concatenate(finite_parts) if finite_parts else np.asarray([0.0, 1.0])
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            if metric == "selection_frequency":
                vmin, vmax = 0.0, 1.0
            fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4), squeeze=False)
            image = None
            for axis, method, values in zip(axes[0], methods, arrays):
                image = axis.imshow(values, vmin=vmin, vmax=vmax, cmap=color_map, origin="upper")
                axis.set_title(method)
                axis.set_xlabel("Patch column")
                axis.set_ylabel("Patch row")
            fig.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=0.82)
            fig.suptitle("{} | target {} kbps | {}".format(sequence_id, target, label))
            fig.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.82, wspace=0.28)
            safe_sequence = str(sequence_id).replace("/", "_")
            fig.savefig(output_dir / "{}_{}kbps_{}.png".format(safe_sequence, target, metric), dpi=160)
            plt.close(fig)
