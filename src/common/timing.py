import time
from contextlib import contextmanager


@contextmanager
def measured(target, key):
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        target[key] = (time.perf_counter_ns() - started) / 1e9
