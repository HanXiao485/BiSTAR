import numpy as np


def residual_difference(ref_previous, ref_current, rec_previous, rec_current):
    a = ref_current.astype(float) - ref_previous.astype(float); b = rec_current.astype(float) - rec_previous.astype(float)
    return float(np.mean(np.abs(a - b)) / 255.0)
