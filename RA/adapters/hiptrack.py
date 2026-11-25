"""HIPTrack (Cai et al., CVPR 2024), official code: https://github.com/WenRuiCai/HIPTrack
Weights: the official ``hiptrack_all_data.pth.tar`` or a fine-tuned checkpoint of the same architecture.

Confidence = peak of the raw centre score map of ``network.forward_track``. The snapshot covers the box, the frame
counter and the HIP memory bank of the decoder; the memory bank is reset at every sequence initialisation (the
official HIPTrack test creates a new tracker per sequence).
"""
from types import SimpleNamespace
from ._common import setup_repo, resolve
from . import compat

DEFAULT_CONFIG = 'experiments/hiptrack/hiptrack.yaml'
MEM = ('mem_k', 'mem_v', 'mem_cnt', 'imgs', 'temp_k', 'temp_v')


class HIPTrackAdapter:
    color = 'RGB'
    def __init__(self, repo, checkpoint, config=None):
        repo = setup_repo(repo)
        compat.install_all()
        from lib.config.hiptrack.config import cfg, update_config_from_file
        update_config_from_file(str(resolve(repo, config or DEFAULT_CONFIG)))
        self.checkpoint = str(resolve(repo, checkpoint))
        params = SimpleNamespace(cfg=cfg, checkpoint=self.checkpoint, debug=0, save_all_boxes=False,
                                 template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
                                 search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE)
        from lib.test.tracker.hiptrack import HIPTrack
        self.t = HIPTrack(params, 'uav_custom', visualize_during_infer=False)
        self.dec = self.t.network.HIP.decoder
        orig = self.t.network.forward_track
        def forward_track(*fa, **fk):
            out = orig(*fa, **fk); self.score = float(out['score_map'].max()); return out
        self.t.network.forward_track = forward_track
    def init(self, image, box, seq):
        self.t.network.set_eval()            # fresh HIP memory bank per sequence (official per-sequence tracker)
        self.t.initialize(image, {'init_bbox': box}, seq)
    def step(self, image): return self.t.track(image)['target_bbox'], self.score
    def state(self): return list(self.t.state)
    def snap(self): return dict(state=list(self.t.state), frame_id=self.t.frame_id, **{k: getattr(self.dec, k, None) for k in MEM})
    def restore(self, s):
        self.t.state = list(s['state']); self.t.frame_id = s['frame_id']
        for k in MEM: setattr(self.dec, k, s[k])
    def on_direct_reset(self): self.t.frame_id += 1   # a direct reset counts as one tracked frame
    def set_box(self, b): self.t.state = [float(v) for v in b]
