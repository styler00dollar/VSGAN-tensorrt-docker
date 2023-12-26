# todo: update script
import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
import functools
from src.scene_detect import scene_detect
from vsgmfss_fortuna import gmfss_fortuna
import torch

torch.set_float32_matmul_precision("medium")

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 8

core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")
core.std.LoadPlugin(path="/usr/local/lib/x86_64-linux-gnu/libawarpsharp2.so")


def vfi_frame_merger(
    clip1: vs.VideoNode,
    clip2: vs.VideoNode,
) -> vs.VideoNode:
    core = vs.core
    metric_thresh = 0.999

    def execute(n: int, clip1: vs.VideoNode, clip2: vs.VideoNode) -> vs.VideoNode:
        try:
            ssim_clip = clip1.get_frame(n).props.get("float_ssim")
            if (ssim_clip and ssim_clip > metric_thresh) or clip1.get_frame(
                n
            ).props.get("_SceneChangeNext"):
                return clip1
            return clip2
        except Exception:
            return clip1

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip1, width=clip1.width, height=clip1.height),
        functools.partial(execute, clip1=clip1, clip2=clip2),
    )


def upscale_frame_skip(n, upscaled, metric_thresh, f):
    ssim_clip = f.props.get("float_ssim")

    if n == 0 or n == len(upscaled) - 1 or (ssim_clip and ssim_clip > metric_thresh):
        return upscaled
    else:
        return upscaled[n - 1]


def metrics_func(clip):
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    return core.vmaf.Metric(clip, offs1, 2)


def inference_clip(video_path="", clip=None):
    # clip = core.ffms2.Source(source=video_path, cache=False)

    clip = core.lsmas.LWLibavSource(source=video_path)
    clip = metrics_func(clip)

    # scene detect
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")
    clip_orig = scene_detect(clip, model_name="efficientnetv2b0+rife46", thresh=0.98)
    clip_orig = vs.core.std.Interleave([clip_orig] * 4)

    # interp
    clip = gmfss_fortuna(
        clip,
        num_streams=2,
        trt=True,
        factor_num=4,
        factor_den=1,
        model=1,
        ensemble=False,
        sc=False,
        trt_cache_path="/workspace/tensorrt/",
    )
    clip = vfi_frame_merger(clip_orig, clip)

    # upscale
    upscaled = core.trt.Model(
        clip,
        engine_path="/workspace/tensorrt/cugan_pro-denoise3x-up2x_fp16_opset18_clamp_and_colorfix.engine",
        num_streams=2,
    )

    upscaled = vs.core.resize.Bicubic(upscaled, format=vs.YUV420P10, matrix_s="709")
    upscaled = metrics_func(upscaled)
    partial = functools.partial(
        upscale_frame_skip, upscaled=upscaled, metric_thresh=0.999
    )
    clip = core.std.FrameEval(
        core.std.BlankClip(clip=upscaled, width=upscaled.width, height=upscaled.height),
        partial,
        prop_src=[upscaled],
    )

    # sharpening
    clip = core.warp.AWarpSharp2(
        clip,
        thresh=128,
        blur=2,
        type=0,
        depth=[16, 8, 8],
        chroma=0,
        opt=True,
        planes=[0, 1, 2],
        cplace="mpeg1",
    )
    clip = core.cas.CAS(clip, sharpness=0.5)

    return clip
