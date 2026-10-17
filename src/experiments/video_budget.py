class VideoSaveLimiter:
    """Limit optional reconstruction-video artifacts without changing evaluation."""

    def __init__(self, limit=None):
        self.limit = None if limit is None else max(0, int(limit))
        self.saved = 0

    def acquire(self):
        if self.limit is not None and self.saved >= self.limit:
            return False
        self.saved += 1
        return True

    @property
    def remaining(self):
        return None if self.limit is None else max(0, self.limit - self.saved)
