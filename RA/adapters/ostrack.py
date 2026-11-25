"""OSTrack-384 (Ye et al., ECCV 2022), official code: https://github.com/botaoye/OSTrack
Config ``vitb_384_mae_ce_32x4_ep300``; the paper used the ModelScope release of these weights
(``damo/cv_vitb_video-single-object-tracking_ostrack``). Confidence = peak of the box-head score map."""
from pathlib import Path
from ._common import setup_repo, resolve
from . import compat

DEFAULT_CONFIG = 'experiments/ostrack/vitb_384_mae_ce_32x4_ep300.yaml'


class OSTrackAdapter:
    color = 'RGB'
    def __init__(self, repo, checkpoint, config=None):
        root = setup_repo(repo)
        self.checkpoint = str(resolve(root, checkpoint))
        self.a = compat.build_ostrack_factory(root, resolve(root, config or DEFAULT_CONFIG), self.checkpoint)()
        self.t = self.a.tracker
        # the builder installs the legacy shims itself (calling compat.install_all as well breaks the jpeg4py shim)
    def init(self, image, box, seq): self.t.initialize(image, {'init_bbox': box})
    def step(self, image): return list(self.t.track(image)['target_bbox']), float(self.a.score)
    def state(self): return list(self.t.state)
    def snap(self): return (list(self.t.state), self.t.frame_id)
    def restore(self, s): self.t.state, self.t.frame_id = list(s[0]), s[1]
    def set_box(self, b): self.t.state = [float(v) for v in b]
