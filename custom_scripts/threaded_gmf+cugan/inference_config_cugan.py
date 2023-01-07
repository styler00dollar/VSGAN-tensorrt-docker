import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
from src.dedup import get_dedup_frames

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 4
core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")

clip = core.ffms2.Source(
    source=globals()["source"], fpsnum=24000 * 2, fpsden=1001, cache=False
)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

frames_duplicated, frames_duplicating = get_dedup_frames()
clip = core.std.DeleteFrames(clip, frames_duplicated)
clip = core.trt.Model(
    clip,
    engine_path="/workspace/tensorrt/cugan.engine",
    num_streams=4,
)
clip = core.std.DuplicateFrames(clip, frames_duplicating)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
