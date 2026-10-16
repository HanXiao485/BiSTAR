import copy
import hashlib
import itertools
import json


METHODS = ("h264", "h265", "bistar", "bistar_atlas")
OBJECTIVES = {"roi_ssim": True, "roi_psnr": True, "critical_patch_retention": True,
              "roi_normalized_l1": False}


def expand_candidates(methods, load_config):
    candidates = []
    for method in METHODS:
        if method not in methods:
            continue
        options = methods.get(method, {})
        if not options.get("enabled", True):
            continue
        base = load_config(options["config"])
        grid = options.get("grid", {})
        paths = list(grid)
        if any(not isinstance(grid[path], list) or not grid[path] for path in paths):
            raise ValueError("Search grid entries must be nonempty lists")
        for values in itertools.product(*(grid[path] for path in paths)):
            config = copy.deepcopy(base)
            for path, value in zip(paths, values):
                node = config["codec"]
                parts = path.split(".")
                for part in parts[:-1]:
                    if part not in node or not isinstance(node[part], dict):
                        raise ValueError("Unknown search grid path: {}".format(path))
                    node = node[part]
                if parts[-1] not in node and path not in ("selector.patch_budget_mode", "encoder.pixel_format", "gop_size"):
                    raise ValueError("Unknown search grid field: {}".format(path))
                node[parts[-1]] = value
            encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
            candidate_id = method + "_" + hashlib.sha256(encoded.encode()).hexdigest()[:12]
            if any(item["candidate_id"] == candidate_id for item in candidates):
                continue
            candidates.append({"method": method, "candidate_id": candidate_id,
                               "config": config, "overrides": dict(zip(paths, values))})
    return candidates


def choose_winners(rows, objective="roi_ssim"):
    if objective not in OBJECTIVES:
        raise ValueError("Unsupported task objective: {}".format(objective))
    winners = []
    for target, method in sorted({(row["target_kbps"], row["method"]) for row in rows}):
        current = [row for row in rows if row["target_kbps"] == target and row["method"] == method]
        candidates = sorted({row["candidate_id"] for row in current})
        complete = []
        required_sequences = {row["sequence_id"] for row in current}
        for candidate in candidates:
            trials = [row for row in current if row["candidate_id"] == candidate]
            if {row["sequence_id"] for row in trials} != required_sequences:
                continue
            if any(row["status"] != "matched" or row.get(objective) is None for row in trials):
                continue
            score = sum(row[objective] for row in trials) / len(trials)
            error = sum(abs(row["rate_error_percent"]) for row in trials) / len(trials)
            complete.append((score, error, candidate))
        if complete:
            best = min(complete, key=lambda item: (-item[0] if OBJECTIVES[objective] else item[0], item[1], item[2]))
            winners.append({"target_kbps": target, "method": method, "candidate_id": best[2],
                            "objective": objective, "calibration_score": best[0]})
    return winners


def compare_validation(rows, objective, pair_tolerance_percent):
    comparisons = []
    for row in rows:
        if row["method"] not in ("bistar", "bistar_atlas"):
            continue
        for baseline in ("h264", "h265"):
            other = next((item for item in rows if item["sequence_id"] == row["sequence_id"]
                          and item["target_kbps"] == row["target_kbps"] and item["method"] == baseline), None)
            comparable = other is not None and row["status"] == "matched" and other["status"] == "matched"
            rate_gap = None if other is None else abs(row["actual_bitrate_kbps"] - other["actual_bitrate_kbps"]) / row["target_kbps"] * 100
            comparable = comparable and rate_gap <= pair_tolerance_percent
            delta = None if not comparable else row[objective] - other[objective]
            wins = delta is not None and (delta > 0 if OBJECTIVES[objective] else delta < 0)
            comparisons.append({"sequence_id": row["sequence_id"], "target_kbps": row["target_kbps"],
                                "method": row["method"], "baseline": baseline,
                                "candidate_id": row["candidate_id"],
                                "baseline_candidate_id": other["candidate_id"] if other else None,
                                "objective": objective, "delta": delta,
                                "pair_rate_gap_percent": rate_gap,
                                "comparable": bool(comparable), "outperforms": bool(wins)})
    return comparisons
