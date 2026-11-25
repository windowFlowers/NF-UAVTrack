"""TCTrack++ (Cao et al., TPAMI 2023), official code: https://github.com/vision4robotics/TCTrack
Config ``experiments/TCTrack++/config.yaml`` with the official TCTrack++ weights; fixed OTB100 offline hyper-parameters
for all datasets. Input BGR. Confidence = ``best_score``. The snapshot includes the temporal memory and feature sets."""
import numpy as np
import torch
from ._common import setup_repo, resolve
from . import compat

DEFAULT_CONFIG = 'experiments/TCTrack++/config.yaml'


class TCTrackAdapter:
    color = 'BGR'
    def __init__(self, repo, checkpoint, config=None):
        repo = setup_repo(repo)
        compat.install_all()
        from pysot.core.config import cfg
        from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
        from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
        cfg.merge_from_file(str(resolve(repo, config or DEFAULT_CONFIG)))
        model = ModelBuilder_tctrackplus('test')
        self.checkpoint = str(resolve(repo, checkpoint))
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.t = TCTrackplusTracker(model.cuda().eval())
        self.hp = list(cfg.HP_SEARCH_TCTrackpp_offline.OTB100)
    def init(self, image, box, seq): self.t.init(image, box)
    def step(self, image):
        out = self.t.track(image, self.hp)
        return list(out['bbox']), float(out['best_score'])
    def state(self):
        c, s = self.t.center_pos, self.t.size
        return [float(c[0] - s[0] / 2), float(c[1] - s[1] / 2), float(s[0]), float(s[1])]
    def snap(self):
        m = self.t.model
        return (self.t.center_pos.copy(), self.t.size.copy(), m.memory, m.featset1, m.featset2)
    def restore(self, s):
        m = self.t.model
        self.t.center_pos, self.t.size = s[0].copy(), s[1].copy(); m.memory, m.featset1, m.featset2 = s[2], s[3], s[4]
    def set_box(self, b):
        self.t.center_pos = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2]); self.t.size = np.array([b[2], b[3]])
