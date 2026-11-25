"""MixFormerV2-B (Cui et al., NeurIPS 2023), official code: https://github.com/MCG-NJU/MixFormerV2
Config ``mixformer2_vit_online/288_depth8_score`` with the official ``mixformerv2_base.pth.tar``.
UAV online size is 3 (config default): online templates are cached in the network as per-layer ``qkv_mem``, so the
snapshot copies the tracker's own fields plus every attention layer's ``qkv_mem``. Confidence = ``conf_score``."""
import copy
import torch
from ._common import setup_repo, resolve

DEFAULT_CONFIG = 'experiments/mixformer2_vit_online/288_depth8_score.yaml'


class MixFormerAdapter:
    color = 'RGB'
    EXC = ('network', 'preprocessor', 'cfg', 'params')
    def __init__(self, repo, checkpoint, config=None):
        self.copy = copy
        root = setup_repo(repo)
        from lib.config.mixformer2_vit_online.config import update_new_config_from_file
        from lib.test.tracker.mixformer2_vit_online import get_tracker_class
        from lib.test.utils import TrackerParams
        cfg = update_new_config_from_file(str(resolve(root, config or DEFAULT_CONFIG)))
        self.checkpoint = str(resolve(root, checkpoint))
        params = TrackerParams()
        params.cfg = cfg; params.checkpoint = self.checkpoint
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR; params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR; params.search_size = cfg.TEST.SEARCH_SIZE
        params.save_all_boxes = False; params.debug = 0; params.vis_attn = 0; params.device = 'cuda'
        params.debug_contrast_dir = None; params.debug_contrast_every = 1
        params.online_size = cfg.TEST.ONLINE_SIZES.get('UAV', [1])[0]; params.update_interval = cfg.TEST.UPDATE_INTERVALS.get('UAV', [25])[0]
        self.t = get_tracker_class()(params, 'uav')
        self.mem = [m for m in self.t.network.modules() if hasattr(m, 'qkv_mem')]
    def init(self, image, box, seq): self.t.initialize(image, {'init_bbox': box, 'sequence_name': seq})
    def step(self, image):
        out = self.t.track(image, {}); return list(out['target_bbox']), float(out['conf_score'])
    def state(self): return list(self.t.state)
    def _cp(self, v): return v.clone() if torch.is_tensor(v) else self.copy.deepcopy(v)
    def snap(self):
        return ({k: self._cp(v) for k, v in vars(self.t).items() if k not in self.EXC and not isinstance(v, torch.nn.Module)},
                [self._cp(m.qkv_mem) for m in self.mem])
    def restore(self, s):
        for k, v in s[0].items(): setattr(self.t, k, self._cp(v))
        for m, v in zip(self.mem, s[1]): m.qkv_mem = self._cp(v)
    def set_box(self, b): self.t.state = [float(v) for v in b]
