import csv


def select_columns(rows, columns): return [{key: row.get(key) for key in columns} for row in rows]


def write_table(rows, columns, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(select_columns(rows, columns))


def write_rd_table(rows, path):
    write_table(rows, ["method", "target_kbps", "actual_bitrate_kbps", "psnr", "ssim", "ms_ssim", "lpips", "status"], path)


def write_runtime_table(rows, path):
    write_table(rows, ["method", "encode_time_sec", "decode_time_sec", "num_frames"], path)


def write_ablation_table(rows, path):
    write_table(rows, ["transport", "selector", "visible_patch_count", "window_size",
                       "encoded_area_fraction", "actual_bitrate_kbps", "ssim", "lpips",
                       "roi_ssim", "roi_lpips", "temporal_error", "reconstruction_ms", "fps"], path)
