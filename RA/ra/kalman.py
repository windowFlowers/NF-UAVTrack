"""Constant-velocity Kalman motion model used by RA for prediction and gating.

The noise multipliers were fitted by maximum likelihood on correct-tracking segments of the training pool
(source-disjoint from the test sequences); they are read from ``configs/chi2_gate_fit.json``.
"""
import json, math
from pathlib import Path
import numpy as np

FIT_FILE = Path(__file__).resolve().parent.parent / 'configs' / 'chi2_gate_fit.json'
MCC2_THRESHOLD = 135.856   # gamma: 99.9% quantile of training normal-frame Mahalanobis d2


class FittedKF:
    # CV Kalman, MLE-fitted noise multipliers
    W_POS, W_VEL = 1 / 20, 1 / 160
    _f = json.load(open(FIT_FILE))['fitted_multipliers']
    A, B, R = _f['pos'], _f['vel'], _f['meas']
    def __init__(self, box):
        cx, cy, s = box[0] + box[2] / 2, box[1] + box[3] / 2, math.sqrt(box[2] * box[3])
        self.x = np.array([cx, cy, 0., 0.])
        self.P = np.diag([(2 * self.A * self.W_POS * s) ** 2] * 2 + [(10 * self.B * self.W_VEL * s) ** 2] * 2)
    def predict(self, dt, s):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        Q = np.diag([(self.A * self.W_POS * s) ** 2] * 2 + [(self.B * self.W_VEL * s) ** 2] * 2) * dt
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q
        self.S = self.P[:2, :2] + np.eye(2) * (self.R * self.W_POS * s) ** 2
        return self.x[:2].copy()
    def update(self, z, s):
        S = self.P[:2, :2] + np.eye(2) * (self.R * self.W_POS * s) ** 2
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - self.x[:2]); self.P = self.P - K @ self.P[:2, :]
