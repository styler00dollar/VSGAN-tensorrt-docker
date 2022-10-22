# https://github.com/ckkelvinchan/RealBasicVSR/blob/master/realbasicvsr/models/builder.py
from mmcv import build_from_cfg
from mmedit.models.registry import BACKBONES, COMPONENTS, LOSSES, MODELS
import functools
import math
import mmcv
import numpy as np
import os
import torch
import torch.nn as nn
import vapoursynth as vs
from .download import check_and_download


def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)

    return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return build(cfg, BACKBONES)


def build_component(cfg):
    """Build component.

    Args:
        cfg (dict): Configuration for building component.
    """
    return build(cfg, COMPONENTS)


def build_loss(cfg):
    """Build loss.

    Args:
        cfg (dict): Configuration for building loss.
    """
    return build(cfg, LOSSES)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        test_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


# https://github.com/HolyWu/vs-basicvsrpp
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4


class realbasicvsr_inference:
    def __init__(self, fp16=True):
        self.fp16 = fp16
        self.scale = 4
        self.cache = True

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # https://github.com/ckkelvinchan/RealBasicVSR/blob/master/inference_realbasicvsr.py
        config = mmcv.Config.fromfile("/workspace/tensorrt/src/realbasicvsr_config.py")
        config.model.pretrained = None
        config.test_cfg.metrics = None
        self.model = build_model(config.model, test_cfg=config.test_cfg)
        model_path = "/workspace/tensorrt/models/RealBasicVSR_x4.pth"
        check_and_download(model_path)
        self.model.load_state_dict(
            torch.load(model_path)["state_dict"],
            strict=True,
        )
        self.model.cuda().eval()

        if self.fp16:
            self.model.half()

    @torch.inference_mode()
    def execute(self, imgs) -> vs.VideoNode:
        if self.fp16:
            imgs = imgs.half()

        output = self.model(imgs.cuda(), test_mode=True)["output"]

        output = output.squeeze(0).detach()

        return output
