import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 4  # can influence ram usage
# only needed if you are inside docker
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


def inference_clip(video_path="", clip=None):
    if clip is None:
        clip = core.lsmas.LWLibavSource(source=video_path)

    # convert colorspace
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

    # vs-mlrt (you need to create the engine yourself, read the readme)
    clip = core.trt.Model(
        clip,
        engine_path="/workspace/tensorrt/cugan_pro-denoise3x-up2x_fp16_opset18_clamp_and_colorfix.engine",
        # tilesize=[854, 480],
        overlap=[0, 0],
        num_streams=4,
    )

    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
