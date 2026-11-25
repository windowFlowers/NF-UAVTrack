"""SiamRPN++ R50 (Li et al., CVPR 2019) and SiamRPN++ LT, official PySOT code: https://github.com/STVIR/pysot

* ``siamrpn``  : config ``experiments/siamrpn_r50_l234_dwxcorr/config.yaml`` with the OpenMMLab (MMTracking)
  UAV123 SiamRPN++ checkpoint, whose key names are mapped to PySOT by ``openmmlab_siamrpn.py``.
  RGB frames are converted to BGR for PySOT.
* ``siamrpnlt``: config ``experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml`` with the official PySOT VOT2018-LT
  model (``SiamRPNLTTracker``). BGR input.
Confidence = ``best_score``.
"""
import cv2
import numpy as np
import torch
from ._common import setup_repo, resolve
from . import openmmlab_siamrpn

DEFAULT_CONFIG = 'experiments/siamrpn_r50_l234_dwxcorr/config.yaml'
DEFAULT_CONFIG_LT = 'experiments/siamrpn_r50_l234_dwxcorr_lt/config.yaml'


def _rgb_to_bgr(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("tracker image must have shape HxWx3")
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _require_bbox(value):
    if value is None or len(value) != 4:
        raise ValueError(f"tracker returned an invalid bbox: {value}")
    bbox = [float(item) for item in value]
    if not np.isfinite(bbox).all() or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ValueError(f"tracker returned an invalid bbox: {value}")
    return bbox


class _PySOTSiamRPNPP:
    """PySOT SiamRPN++ with explicit OpenMMLab checkpoint mapping; RGB in, xywh pixel out."""

    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
        from pysot.core.config import cfg
        from pysot.models.model_builder import ModelBuilder
        from pysot.tracker.tracker_builder import build_tracker

        cfg.merge_from_file(str(config_path))
        cfg.CUDA = self.device.type == "cuda"
        model = ModelBuilder()
        openmmlab_siamrpn.load_openmmlab_checkpoint(model, str(checkpoint_path), map_location="cpu")
        model = model.to(self.device).eval()
        self._tracker = build_tracker(model)

    def initialize(self, image, info):
        bbox = _require_bbox(info.get("init_bbox"))
        self._tracker.init(_rgb_to_bgr(image), bbox)
        return {"target_bbox": bbox, "conf_score": 1.0, "tracking_state": "tracked"}

    def track(self, image, info=None):
        output = self._tracker.track(_rgb_to_bgr(image))
        bbox = _require_bbox(output.get("bbox"))
        score = float(output.get("best_score", 0.0))
        return {"target_bbox": bbox, "conf_score": score, "tracking_state": "tracked" if score > 0.0 else "lost"}


class SiamRPNAdapter:
    color = 'RGB'
    def __init__(self, repo, checkpoint, config=None):
        root = setup_repo(repo)
        self.checkpoint = str(resolve(root, checkpoint))
        self.a = _PySOTSiamRPNPP(str(resolve(root, config or DEFAULT_CONFIG)), self.checkpoint, device='cuda:0')
        self.t = self.a._tracker
    def init(self, image, box, seq): self.a.initialize(image, {'init_bbox': box})
    def step(self, image):
        out = self.a.track(image); return list(out['target_bbox']), float(out['conf_score'])
    def state(self):
        (cx, cy), (w, h) = self.t.center_pos, self.t.size; return [cx - w / 2, cy - h / 2, w, h]
    def snap(self): return (self.t.center_pos.copy(), self.t.size.copy())
    def restore(self, s): self.t.center_pos, self.t.size = s[0].copy(), s[1].copy()
    def set_box(self, b):
        x, y, w, h = [float(v) for v in b]
        self.t.center_pos = np.array([x + (w - 1) / 2, y + (h - 1) / 2]); self.t.size = np.array([w, h])


class SiamRPNLTAdapter:
    color = 'BGR'
    def __init__(self, repo, checkpoint, config=None):
        root = setup_repo(repo)
        from pysot.core.config import cfg
        from pysot.models.model_builder import ModelBuilder
        from pysot.tracker.tracker_builder import build_tracker
        from pysot.utils.model_load import load_pretrain
        from yacs.config import CfgNode
        c = CfgNode.load_cfg(open(resolve(root, config or DEFAULT_CONFIG_LT)))
        def prune(a, b, pre=''):   # drop keys unknown to this pysot version; only channel annotations are expected
            out = []
            for k in list(a):
                if k not in b: a.pop(k); out.append(pre + k)
                elif isinstance(a[k], CfgNode) and not b[k].is_new_allowed(): out += prune(a[k], b[k], pre + k + '.')
            return out
        assert c.RPN.WEIGHTED == c.RPN.KWARGS.weighted
        dropped = prune(c, cfg)
        assert set(dropped) <= {'BACKBONE.CHANNELS', 'ADJUST.ADJUST_CHANNEL', 'RPN.WEIGHTED'}, dropped   # duplicates of KWARGS entries
        print('pysot LT config: dropped informational keys', dropped, flush=True)
        cfg.merge_from_other_cfg(c); cfg.CUDA = True
        self.checkpoint = str(resolve(root, checkpoint))
        self.t = build_tracker(load_pretrain(ModelBuilder(), self.checkpoint).cuda().eval())
        assert type(self.t).__name__ == 'SiamRPNLTTracker', type(self.t)
    def init(self, image, box, seq):
        self.t.init(image, [float(v) for v in box]); self.t.longterm_state = False   # fresh tracker per sequence
    def step(self, image):
        out = self.t.track(image); return [float(v) for v in out['bbox']], float(out['best_score'])
    def state(self):
        (cx, cy), (w, h) = self.t.center_pos, self.t.size; return [cx - w / 2, cy - h / 2, w, h]
    def snap(self): return (self.t.center_pos.copy(), self.t.size.copy(), self.t.longterm_state)
    def restore(self, s): self.t.center_pos, self.t.size, self.t.longterm_state = s[0].copy(), s[1].copy(), s[2]
    def set_box(self, b):
        x, y, w, h = [float(v) for v in b]
        self.t.center_pos = np.array([x + (w - 1) / 2, y + (h - 1) / 2]); self.t.size = np.array([w, h])
