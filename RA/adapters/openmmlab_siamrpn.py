"""Load an OpenMMLab SiamRPN checkpoint into the local PySOT ModelBuilder.

The checkpoint produced by MMTracking uses the same tensor values and model
topology as this PySOT configuration, but names its neck and RPN head modules
using OpenMMLab conventions.  This module keeps that conversion explicit and
does not change PySOT's model loader, tracker, or configuration.
"""

from __future__ import absolute_import, division, print_function

import argparse
import re

import torch


_NECK_KEY = re.compile(r"^neck\.convs\.(\d+)\.(conv|bn)\.(.+)$")
_HEAD_KEY = re.compile(r"^head\.(cls_heads|reg_heads)\.(\d+)\.(.+)$")


def _load_checkpoint(path, map_location="cpu"):
    """Load tensor-only checkpoints when supported by the installed PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # ``weights_only`` is unavailable in the old PyTorch versions used by
        # the original PySOT environment.
        return torch.load(path, map_location=map_location)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a mapping or contain a state_dict mapping")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint['state_dict'] must be a mapping")
    return state_dict


def _remap_key(key):
    """Translate one MMTracking state-dict key to a PySOT key."""
    if key.startswith("module."):
        key = key[len("module."):]

    if key.startswith("backbone."):
        return key

    if key == "head.cls_weight":
        return "rpn_head.cls_weight"
    if key == "head.reg_weight":
        return "rpn_head.loc_weight"

    match = _NECK_KEY.match(key)
    if match:
        index, block, suffix = match.groups()
        target_block = 0 if block == "conv" else 1
        return "neck.downsample{}.downsample.{}.{}".format(
            int(index) + 2, target_block, suffix)

    match = _HEAD_KEY.match(key)
    if match:
        branch, index, suffix = match.groups()
        stage = int(index) + 2
        target_branch = "cls" if branch == "cls_heads" else "loc"
        target_prefix = "rpn_head.rpn{}.{}".format(stage, target_branch)
        conversions = (
            ("kernel_convs.conv.", "conv_kernel.0."),
            ("kernel_convs.bn.", "conv_kernel.1."),
            ("search_convs.conv.", "conv_search.0."),
            ("search_convs.bn.", "conv_search.1."),
            ("head_convs.0.conv.", "head.0."),
            ("head_convs.0.bn.", "head.1."),
            ("head_convs.1.conv.", "head.3."),
        )
        for source_prefix, target_suffix in conversions:
            if suffix.startswith(source_prefix):
                return target_prefix + "." + target_suffix + suffix[
                    len(source_prefix):]

    raise KeyError("unsupported OpenMMLab checkpoint key: {}".format(key))


def remap_state_dict(state_dict):
    """Return a strictly PySOT-named copy of an OpenMMLab state dict."""
    remapped = {}
    for source_key, value in state_dict.items():
        target_key = _remap_key(source_key)
        if target_key in remapped:
            raise ValueError("duplicate mapped checkpoint key: {}".format(target_key))
        remapped[target_key] = value
    return remapped


def _validate_state_dict(model, state_dict):
    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatch = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(model_state[key].shape)
        checkpoint_shape = tuple(state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatch.append((key, model_shape, checkpoint_shape))

    if missing or unexpected or shape_mismatch:
        details = []
        if missing:
            details.append("missing={}".format(missing))
        if unexpected:
            details.append("unexpected={}".format(unexpected))
        if shape_mismatch:
            details.append("shape_mismatch={}".format(shape_mismatch))
        raise RuntimeError("checkpoint is not compatible with ModelBuilder: " +
                           "; ".join(details))


def load_openmmlab_checkpoint(model, checkpoint_path, map_location="cpu"):
    """Strictly load an OpenMMLab SiamRPN checkpoint into ``model``.

    The caller must merge the matching PySOT config before constructing
    ``ModelBuilder``.  The returned model is the same object passed in.
    """
    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
    source_state = _extract_state_dict(checkpoint)
    remapped_state = remap_state_dict(source_state)
    _validate_state_dict(model, remapped_state)
    model.load_state_dict(remapped_state, strict=True)
    return model


def _run_smoke_forward(model, device):
    template = torch.zeros(1, 3, 127, 127, device=device)
    search = torch.zeros(1, 3, 255, 255, device=device)
    with torch.no_grad():
        model.template(template)
        outputs = model.track(search)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Strictly load an OpenMMLab SiamRPN checkpoint in PySOT")
    parser.add_argument("--config", required=True, help="PySOT YAML config")
    parser.add_argument("--checkpoint", required=True,
                        help="OpenMMLab .pth checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-forward", action="store_true",
                        help="run template/track on zero tensors")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    from pysot.core.config import cfg
    from pysot.models.model_builder import ModelBuilder

    cfg.merge_from_file(args.config)
    model = ModelBuilder()
    load_openmmlab_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model.eval().to(device)

    print("strict_load: PASS")
    print("device: {}".format(device))
    if args.smoke_forward:
        outputs = _run_smoke_forward(model, device)
        print("cls_shape: {}".format(tuple(outputs["cls"].shape)))
        print("loc_shape: {}".format(tuple(outputs["loc"].shape)))
        if hasattr(torch, "isfinite"):
            print("finite: {} {}".format(
                bool(torch.isfinite(outputs["cls"]).all()),
                bool(torch.isfinite(outputs["loc"]).all())))


if __name__ == "__main__":
    main()
