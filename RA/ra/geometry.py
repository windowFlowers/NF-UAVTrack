import numpy as np


def iou(a, b):
    iw = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); ih = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih; u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.


cen = lambda b: np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
