# https://github.com/ckkelvinchan/RealBasicVSR/blob/master/realbasicvsr/models/builder.py
import torch.nn as nn
from mmcv import build_from_cfg
from mmedit.models.registry import BACKBONES, COMPONENTS, LOSSES, MODELS
import functools

def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
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
import math
import os
import numpy as np
import torch
import vapoursynth as vs
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4


def realbasicvsr_model(clip: vs.VideoNode, model: int = 1, interval: int = 15, fp16: bool = False) -> vs.VideoNode:

    scale = 4

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('EGVSR: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('EGVSR: only RGBS format is supported')

    if interval < 1:
        raise vs.Error('EGVSR: interval must be at least 1')

    if not torch.cuda.is_available():
        raise vs.Error('EGVSR: CUDA is not available')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    import mmcv
    # https://github.com/ckkelvinchan/RealBasicVSR/blob/master/inference_realbasicvsr.py
    config = mmcv.Config.fromfile("/workspace/tensorrt/src/realbasicvsr_config.py")
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    model.load_state_dict(torch.load("/workspace/RealBasicVSR_x4.pth")["state_dict"], strict=True)
    model.cuda().eval()

    cache = {}
 
    @torch.inference_mode()
    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        if str(n) not in cache:
            cache.clear()

            imgs = [torch.Tensor(frame_to_tensor(clip.get_frame(n)))]
            for i in range(1, interval):
                if (n + i) >= clip.num_frames:
                    break
                imgs.append(torch.Tensor(frame_to_tensor(clip.get_frame(n + i))))

            imgs = torch.stack(imgs)
            imgs = imgs.unsqueeze(0)
            if fp16:
                imgs = imgs.half()

            output = model(imgs.cuda(), test_mode=True)['output']

            output = output.squeeze(0).detach().cpu().numpy()

            for i in range(output.shape[0]):
                cache[str(n + i)] = output[i, :, :, :]

            del imgs
            torch.cuda.empty_cache()
            
        return tensor_to_clip(clip=clip, image=cache[str(n)])

    return core.std.FrameEval(
            core.std.BlankClip(
                clip=clip,
                width=clip.width * scale,
                height=clip.height * scale
            ),
            functools.partial(
                execute,
                clip=clip
            )
    )

def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    return np.stack([
        np.asarray(frame[plane])
        for plane in range(frame.format.num_planes)
    ])

def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f

def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(
        clip=clip,
        width=image.shape[-1],
        height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )
