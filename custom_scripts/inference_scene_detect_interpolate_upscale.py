import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
import functools
from src.scene_detect import scene_detect
from vsgmfss_fortuna import gmfss_fortuna
from src.vfi_inference import vfi_frame_merger

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 8

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


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
    clip = core.bs.VideoSource(source=video_path)
    clip = metrics_func(clip)

    # scene detect
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")
    clip_orig = scene_detect(
        clip,
        thresh=0.98,
        onnx_path="/workspace/tensorrt/sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx",
        resolution=224,
    )
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
        engine_path="/workspace/tensorrt/2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp.engine",
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

    return clip
