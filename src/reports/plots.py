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
