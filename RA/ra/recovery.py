"""RA controller: trigger -> Kalman prediction / gate -> detector candidate ranking -> tracker re-search -> acceptance.

Per frame:
  1. Pass 1 is the unchanged host-tracker step; its confidence p1 is the tracker's own score.
  2. Trigger: p1 < tau (``trigger='conf'``, paper default) or every ``period`` frames (``trigger='periodic'``, ablation).
  3. Candidates: cached full-frame detections with conf >= kappa0 (``conf_min``), not overlapping the pass-1 box
     (IoU <= 0.3) and with a scale within [0.5, 2] of the last confident box.
  4. Motion gate (variant ``soft``, paper default): while the last confident frame is at most L (``gate_frames``) frames
     old, a candidate with Mahalanobis d2 > gamma (135.856) to the Kalman prediction is kept only if its detector
     confidence >= kappa1, and is ranked after all gated-in candidates. ``hard`` drops it, ``nomo`` ignores motion and
     ranks by detector confidence.
  5. Re-search: for the top-k candidates the tracker state is restored to the pre-pass-1 snapshot, its box is set to
     the candidate and the tracker is stepped again on the same frame. A result is accepted iff s' > p1, s' >= accept
     (= tau by default) and IoU(box', detection) > 0.3; the best accepted result (highest s') becomes the output and
     the tracker state. Otherwise the pass-1 state is restored.
  6. The Kalman filter is updated with confident outputs and re-initialised after an accepted far (d2 > gamma) jump.
Variants ``reset`` / ``reset_nomo`` (ablation: direct detector reset) skip verification and take the top-ranked
candidate as output and state. ``none`` runs the host tracker unchanged. No ground truth is used after initialisation.

Host-tracker interface (``adapters``): init(image, box, seq), step(image) -> (box_xywh, score), state() -> box_xywh,
snap() -> opaque snapshot, restore(snap), set_box(box_xywh); optional on_direct_reset() called after a direct reset.
"""
import math
import numpy as np
from .kalman import FittedKF, MCC2_THRESHOLD
from .geometry import iou, cen

VARIANTS = ('none', 'hard', 'soft', 'nomo', 'reset', 'reset_nomo')


class RA:
    def __init__(self, ad, variant, tau, accept, conf_min=0.25, topk=3, gate_frames=30, trigger='conf', period=15, kappa1=0.5):
        assert variant in VARIANTS, variant
        self.ad, self.variant, self.tau, self.accept = ad, variant, tau, accept
        self.conf_min, self.topk, self.gate_frames = conf_min, topk, gate_frames
        self.trigger, self.period, self.kappa1 = trigger, period, kappa1
    def init(self, image, box, seq):
        self.ad.init(image, box, seq)
        self.kf = FittedKF(box); self.fid = 0; self.kf_last = 0; self.last_conf_frame = 0; self.conf_s = math.sqrt(box[2] * box[3])
    def track(self, image, dets_all):
        self.fid += 1; fid = self.fid
        st = self.ad.state(); s = math.sqrt(st[2] * st[3])
        pred = self.kf.predict(fid - self.kf_last, s); self.kf_last = fid
        snap = self.ad.snap()
        box, p1 = self.ad.step(image); score = p1
        dets = [d for d in dets_all if d[4] >= self.conf_min and d[2] > 0 and d[3] > 0]
        triggered = self.variant != 'none' and ((fid % self.period == 0) if self.trigger == 'periodic' else (p1 < self.tau))
        used = False; m_used = None; tried = 0; ncand = 0
        if triggered and dets:
            recent = fid - self.last_conf_frame <= self.gate_frames
            cands = []
            for d in dets:
                if iou(d[:4], box) > 0.3 or not 0.5 <= math.sqrt(d[2] * d[3]) / self.conf_s <= 2.0:
                    continue
                v = cen(d) - pred; m = float(v @ np.linalg.solve(self.kf.S, v))
                if self.variant == 'hard':
                    if recent and m > MCC2_THRESHOLD: continue
                    cands.append((m, d, m))
                elif self.variant in ('soft', 'reset'):
                    if recent and m > MCC2_THRESHOLD:
                        if d[4] < self.kappa1: continue
                        cands.append((1e9 - d[4], d, m))
                    else:
                        cands.append((m, d, m))
                else:   # nomo / reset_nomo: no motion gate, ranked by detector confidence
                    cands.append((-d[4], d, m))
            cands.sort(key=lambda c: c[0]); ncand = len(cands)
            if cands and self.variant.startswith('reset'):
                # direct detector reset (ablation): top-ranked admissible detection becomes output and state, no verification
                _, d, m = cands[0]; self.ad.restore(snap)
                if hasattr(self.ad, 'on_direct_reset'): self.ad.on_direct_reset()
                self.ad.set_box(d[:4])
                box = [float(v) for v in d[:4]]; score = max(self.tau, self.accept); m_used = m; used = True
            elif cands:
                first = (self.ad.snap(), box, p1); best = None
                for _, d, m in cands[:self.topk]:
                    self.ad.restore(snap); self.ad.set_box(d[:4])
                    b2, s2 = self.ad.step(image); tried += 1
                    if s2 > p1 and s2 >= self.accept and iou(b2, d[:4]) > 0.3 and (best is None or s2 > best[2]):
                        best = (self.ad.snap(), b2, s2, m)
                if best is not None:
                    self.ad.restore(best[0]); box, score, m_used, used = best[1], best[2], best[3], True
                else:
                    self.ad.restore(first[0]); box = first[1]
        sb = math.sqrt(box[2] * box[3])
        if used and m_used > MCC2_THRESHOLD:
            self.kf = FittedKF(box)
        elif score >= self.tau:
            self.kf.update(cen(box), sb)
        if score >= self.tau:
            self.last_conf_frame = fid; self.conf_s = sb
        return box, dict(p1=p1, score=score, trig=triggered, used=used, tried=tried, ncand=ncand, ndet=len(dets))
