import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
import functools
from src.scene_detect import scene_detect
from src.vfi_inference import vfi_frame_merger
from src.rife_trt import rife_trt

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
        onnx_path="/workspace/tensorrt/sc_efficientformerv2_s0+rife46_flow_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx",
        resolution=224,
    )
    clip_orig = vs.core.std.Interleave([clip_orig] * 4)

    # interp
    clip = rife_trt(
        clip,
        multi=4,
        scale=1.0,
        device_id=0,
        num_streams=2,
        engine_path="/workspace/tensorrt/rife414_ensembleTrue_op18_fp16_clamp_sim.engine",
    )

    clip = vfi_frame_merger(clip_orig, clip)

    # upscale
    upscaled = core.trt.Model(
        clip,
        engine_path="/workspace/tensorrt/2x_AnimeJaNai_V3_SmoothRC21_Compact_50k_op18_fp16_clamp.engine",
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
