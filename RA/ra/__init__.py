"""Recovery Augmentation (RA): motion-gated detector re-acquisition for single-object UAV trackers.

RA wraps any tracker that exposes a black-box step / snapshot / restore / set-box interface (see ``adapters``).
"""
from .kalman import FittedKF, MCC2_THRESHOLD
from .geometry import iou, cen
from .recovery import RA

__all__ = ['FittedKF', 'MCC2_THRESHOLD', 'iou', 'cen', 'RA']
