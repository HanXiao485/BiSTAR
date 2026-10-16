class PatchBudgetController:
    """Causal GOP-level patch budget controller."""

    def __init__(self, selector_config, reference_kbps, patch_count, target_kbps=None,
                 force_fixed=False):
        self.config = selector_config
        self.fixed = bool(selector_config.get("patch_budget_mode", True)) or force_fixed
        self.reference_budget = int(selector_config["num_visible_patches"])
        self.minimum = max(1, int(selector_config.get("min_visible_patches", 1)))
        self.maximum = min(patch_count, int(selector_config.get("max_visible_patches", patch_count)))
        if self.minimum > self.maximum:
            raise ValueError("min_visible_patches cannot exceed max_visible_patches")
        candidates = selector_config.get("candidate_visible_patches", [])
        self.candidates = sorted({int(value) for value in candidates
                                  if self.minimum <= int(value) <= self.maximum})
        self.alpha = float(selector_config.get("adaptation_alpha", 0.5))
        self.max_step = max(1, int(selector_config.get("max_patch_step", patch_count)))
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("adaptation_alpha must be in (0, 1]")
        reference_kbps = max(float(reference_kbps), 1e-6)
        target_kbps = reference_kbps if target_kbps is None else float(target_kbps)
        initial = self.reference_budget if self.fixed else self.reference_budget * target_kbps / reference_kbps
        if self.fixed:
            if not 1 <= self.reference_budget <= patch_count:
                raise ValueError("Fixed num_visible_patches must be in [1, patch_count]")
            self.current = self.reference_budget
        else:
            self.current = self._quantize(initial)

    @property
    def strategy(self):
        return "fixed" if self.fixed else "adaptive_gop"

    def update(self, total_bytes, valid_frames, fps, target_kbps):
        if self.fixed or total_bytes <= 0:
            return self.current
        target_bytes = float(target_kbps) * 1000.0 / 8.0 * float(valid_frames) / float(fps)
        estimated = self.current * target_bytes / float(total_bytes)
        blended = self.current + self.alpha * (estimated - self.current)
        lower = self.current - self.max_step
        upper = self.current + self.max_step
        candidates = [candidate for candidate in self.candidates if lower <= candidate <= upper]
        bounded = min(self.maximum, max(self.minimum, int(round(min(upper, max(lower, blended))))))
        if self.candidates:
            self.current = min(candidates, key=lambda candidate: (abs(candidate - bounded), candidate)) if candidates else self.current
        else:
            self.current = bounded
        return self.current

    def _quantize(self, value):
        value = min(self.maximum, max(self.minimum, int(round(value))))
        if not self.candidates:
            return value
        return min(self.candidates, key=lambda candidate: (abs(candidate - value), candidate))
