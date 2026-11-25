"""STARK-ST50 (Yan et al., ICCV 2021), official code: https://github.com/researchmm/Stark
Config ``stark_st2/baseline`` with the official ``STARKST_ep0050.pth.tar``; UAV update interval 25.
Confidence = the tracker's own ``conf_score``. The snapshot includes the dynamic-template list, which ``track()``
may replace on update frames."""
from ._common import setup_repo, resolve
from . import compat

DEFAULT_CONFIG = 'experiments/stark_st2/baseline.yaml'


class STARKAdapter:
    color = 'RGB'
    def __init__(self, repo, checkpoint, config=None):
        root = setup_repo(repo)
        self.checkpoint = str(resolve(root, checkpoint))
        self.t = compat.build_stark_factory(root, resolve(root, config or DEFAULT_CONFIG), self.checkpoint)()
    def init(self, image, box, seq): self.t.initialize(image, {'init_bbox': box})
    def step(self, image):
        out = self.t.track(image); return list(out['target_bbox']), float(out['conf_score'])
    def state(self): return list(self.t.state)
    def snap(self): return (list(self.t.state), self.t.frame_id, list(self.t.z_dict_list))
    def restore(self, s): self.t.state, self.t.frame_id, self.t.z_dict_list = list(s[0]), s[1], list(s[2])
    def set_box(self, b): self.t.state = [float(v) for v in b]
