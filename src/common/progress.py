import sys
import time


class ProgressBar:
    """Small dependency-free terminal progress bar for long experiments."""

    def __init__(self, total, description="Progress", width=28):
        self.total = max(0, int(total))
        self.description = str(description)
        self.width = max(10, int(width))
        self.current = 0
        self.started = time.time()
        self._last_render = ""

    def update(self, current=None, message=""):
        self.current = self.current + 1 if current is None else min(self.total, int(current))
        elapsed = max(0.0, time.time() - self.started)
        fraction = 1.0 if self.total == 0 else self.current / float(self.total)
        filled = int(self.width * fraction)
        bar = "=" * filled + ">" + " " * max(0, self.width - filled - 1)
        rate = self.current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.current) / rate if rate > 0 else 0.0
        suffix = " {}".format(message) if message else ""
        text = "\r{} [{}] {}/{} ({:5.1f}%) ETA {:6.1f}s{}".format(
            self.description, bar, self.current, self.total, fraction * 100.0, eta, suffix)
        self._last_render = text
        sys.stdout.write(text)
        sys.stdout.flush()

    def close(self):
        if self.current < self.total:
            self.current = self.total
            self.update(self.current)
        sys.stdout.write("\n")
        sys.stdout.flush()
