import sys
import os

sys.path.append("/workspace/tensorrt/")

import vapoursynth as vs
import functools
from src.scene_detect import scene_detect
from vsgmfss_union import gmfss_union

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 16
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
        ssim_clip = clip1.get_frame(n).props.get("float_ssim")
        if (ssim_clip and ssim_clip > metric_thresh) or clip1.get_frame(n).props.get(
            "_SceneChangeNext"
        ):
            return clip1
        return clip2

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip1, width=clip1.width, height=clip1.height),
        functools.partial(execute, clip1=clip1, clip2=clip2),
    )


def inference_clip(video_path="", clip=None):
    # clip = core.ffms2.Source(source=video_path, cache=False)
    clip = core.lsmas.LWLibavSource(source=video_path)

    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    clip = core.vmaf.Metric(clip, offs1, 2)

    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

    clip_orig = scene_detect(clip, model_name="efficientnetv2_b0", thresh=0.98)
    clip_orig = vs.core.std.Interleave([clip_orig] * 2)

    clip = gmfss_union(
        clip,
        num_streams=4,
        trt=True,
        factor_num=2,
        ensemble=False,
        sc=False,
        trt_cache_path="/workspace/tensorrt/",
    )
    clip = vfi_frame_merger(clip_orig, clip)
    clip = core.trt.Model(
        clip,
        engine_path="/workspace/tensorrt/sudo_shuffle_cugan_fp16_op18_clamped_9.584.969.engine",
        num_streams=4,
    )

    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    clip = core.warp.AWarpSharp2(
        clip,
        thresh=128,
        blur=2,
        type=0,
        depth=[16 * 3, 8 * 3, 8 * 3],
        chroma=0,
        opt=True,
        planes=[0, 1, 2],
        cplace="mpeg1",
    )
    return clip
