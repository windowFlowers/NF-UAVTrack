"""Host-tracker adapters. Each wraps the tracker class of the *official* repository (cloned by the user) behind the
black-box interface RA needs: init / step -> (box, score) / state / snap / restore / set_box.
Imports are lazy so that only the selected tracker's repository has to be present."""
import importlib

ADAPTERS = {
    'hiptrack': ('adapters.hiptrack', 'HIPTrackAdapter'),
    'ostrack': ('adapters.ostrack', 'OSTrackAdapter'),
    'stark': ('adapters.stark', 'STARKAdapter'),
    'ortrack': ('adapters.ortrack', 'ORTrackAdapter'),
    'mixformer': ('adapters.mixformerv2', 'MixFormerAdapter'),
    'tctrackpp': ('adapters.tctrackpp', 'TCTrackAdapter'),
    'siamrpn': ('adapters.siamrpnpp', 'SiamRPNAdapter'),
    'siamrpnlt': ('adapters.siamrpnpp', 'SiamRPNLTAdapter'),
}


def build_adapter(name, repo, checkpoint, config=None):
    mod, cls = ADAPTERS[name]
    return getattr(importlib.import_module(mod), cls)(repo, checkpoint, config)
