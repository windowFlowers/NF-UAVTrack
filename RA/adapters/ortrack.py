"""ORTrack (Wu et al., CVPR 2025), official code: https://github.com/wuyou3474/ORTrack
Weights: official DeiT-tiny checkpoint (``ortrack_deit_tiny.pth.tar``). Confidence = peak of the score map."""
from types import SimpleNamespace
from ._common import setup_repo, resolve
from . import compat

DEFAULT_CONFIG = 'experiments/ortrack/deit_tiny_patch16_224.yaml'


class ORTrackAdapter:
    color = 'RGB'
    def __init__(self, repo, checkpoint, config=None):
        repo = setup_repo(repo)
        compat.install_all()
        from lib.config.ortrack.config import cfg, update_config_from_file
        update_config_from_file(str(resolve(repo, config or DEFAULT_CONFIG)))
        params = SimpleNamespace(cfg=cfg, checkpoint=str(resolve(repo, checkpoint)), debug=0, save_all_boxes=False,
                                 template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
                                 search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE)
        from lib.test.tracker.ortrack import ORTrack
        self.t = ORTrack(params, 'uav_custom')
        orig = self.t.network.forward
        def forward(*a, **k):
            out = orig(*a, **k); self.score = float(out['score_map'].max()); return out
        self.t.network.forward = forward
        self.checkpoint = params.checkpoint
    def init(self, image, box, seq): self.t.initialize(image, {'init_bbox': box})
    def step(self, image): return list(self.t.track(image)['target_bbox']), self.score
    def state(self): return list(self.t.state)
    def snap(self): return (list(self.t.state), self.t.frame_id)
    def restore(self, s): self.t.state, self.t.frame_id = list(s[0]), s[1]
    def set_box(self, b): self.t.state = [float(v) for v in b]
