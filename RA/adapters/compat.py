"""Import shims that let the 2021-2024 official tracker repositories run on a current PyTorch / torchvision stack,
and the OSTrack / STARK tracker factories used for the paper runs. Nothing here changes tracker numerics:
the shims only replace optional imports (visdom, jpeg4py, matplotlib plotting, torch._six, torchvision.models.utils,
lib.train.admin.local) and default torch.load to weights_only=False for legacy checkpoints."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _install_legacy_torch_loader():
    import torch

    original_load = torch.load
    if getattr(original_load, "_antiuav_legacy_compatible", False):
        return

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load._antiuav_legacy_compatible = True
    torch.load = load


def _install_torch_six_shim():
    """Provide the tiny torch._six surface used by the legacy baseline loaders."""
    import torch
    import types

    module_name = "torch._six"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.string_classes = (str,)
    module.int_classes = (int,)
    sys.modules[module_name] = module
    setattr(torch, "_six", module)


def _install_jpeg4py_shim():
    """Let legacy data modules import without requiring the optional JPEG4Py binary."""
    import types

    if importlib.util.find_spec("jpeg4py") is not None or "jpeg4py" in sys.modules:
        return

    module = types.ModuleType("jpeg4py")

    class JPEG:
        def __init__(self, path):
            self.path = str(path)

        def decode(self):
            import cv2

            image = cv2.imread(self.path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {self.path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    JPEG.__module__ = "jpeg4py"
    module.JPEG = JPEG
    sys.modules["jpeg4py"] = module


def _install_visdom_shim():
    """Satisfy OSTrack's optional debug import when Visdom is not installed."""
    if importlib.util.find_spec("visdom") is not None or "visdom" in sys.modules:
        return
    import types

    module = types.ModuleType("visdom")
    module.__path__ = []

    class Visdom:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    Visdom.__module__ = "visdom"
    module.Visdom = Visdom
    server = types.ModuleType("visdom.server")
    module.server = server
    sys.modules["visdom"] = module
    sys.modules["visdom.server"] = server


def _install_torchvision_utils_shim():
    """Restore the torchvision.models.utils import used by the 2021 STARK code."""
    import types
    from torch.hub import load_state_dict_from_url

    module_name = "torchvision.models.utils"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.load_state_dict_from_url = load_state_dict_from_url
    sys.modules[module_name] = module


def _install_ostrack_plotting_shim():
    """Avoid importing an optional matplotlib binary during headless runs."""
    import types

    module_name = "lib.vis.plotting"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.overlay_mask = lambda image, mask, *args, **kwargs: image
    module.show_image_with_boxes = lambda image, boxes, *args, **kwargs: image
    sys.modules[module_name] = module


def _install_train_local_shim():
    import types

    module_name = "lib.train.admin.local"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class EnvironmentSettings:
        pass

    EnvironmentSettings.__module__ = module_name
    module.EnvironmentSettings = EnvironmentSettings
    sys.modules[module_name] = module


class _ConfidenceAdapter:
    def __init__(self, tracker):
        object.__setattr__(self, "tracker", tracker)
        object.__setattr__(self, "score", 1.0)

        def capture(_module, _inputs, output):
            score_map = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(score_map, "detach"):
                object.__setattr__(self, "score", float(score_map.detach().max().item()))

        tracker.network.box_head.register_forward_hook(capture)

    @property
    def state(self):
        return self.tracker.state

    @state.setter
    def state(self, value):
        self.tracker.state = value

    def initialize(self, image, info):
        return self.tracker.initialize(image, info)

    def track(self, image, info=None):
        output = self.tracker.track(image, info)
        output["conf_score"] = float(self.score)
        return output

    def __getattr__(self, name):
        return getattr(self.tracker, name)


def build_ostrack_factory(tracker_root, config_path, checkpoint_path):
    _install_torch_six_shim()
    _install_jpeg4py_shim()
    _install_visdom_shim()
    _install_ostrack_plotting_shim()
    _install_legacy_torch_loader()
    from lib.config.ostrack.config import cfg, update_config_from_file
    from lib.test.tracker.ostrack import OSTrack
    from lib.test.utils import TrackerParams

    update_config_from_file(str(config_path))

    def factory():
        params = TrackerParams()
        params.cfg = cfg
        params.checkpoint = str(checkpoint_path)
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE
        params.save_all_boxes = False
        params.debug = 0
        return _ConfidenceAdapter(OSTrack(params, "uav"))

    factory.search_size = int(cfg.TEST.SEARCH_SIZE)
    return factory


def build_stark_factory(tracker_root, config_path, checkpoint_path):
    _install_torch_six_shim()
    _install_jpeg4py_shim()
    _install_torchvision_utils_shim()
    _install_train_local_shim()
    _install_legacy_torch_loader()
    from lib.config.stark_st2.config import cfg, update_config_from_file
    from lib.test.tracker.stark_st import STARK_ST
    from lib.test.utils import TrackerParams

    update_config_from_file(str(config_path))
    cfg.TEST.UPDATE_INTERVALS.UAV = [25]

    def factory():
        params = TrackerParams()
        params.cfg = cfg
        params.checkpoint = str(checkpoint_path)
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE
        params.save_all_boxes = False
        return STARK_ST(params, "uav")

    return factory


def install_all():
    """Same shims, same order, as the evaluation worker's compatibility() used for HIPTrack, ORTrack and TCTrack++."""
    for fn in (_install_legacy_torch_loader, _install_torch_six_shim, _install_jpeg4py_shim, _install_visdom_shim,
               _install_torchvision_utils_shim, _install_ostrack_plotting_shim, _install_train_local_shim):
        fn()
