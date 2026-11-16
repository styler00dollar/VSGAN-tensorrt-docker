import sys

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
from src.scene_detect import scene_detect
from vsrife import rife

core = vs.core
core.num_threads = 8


def inference_clip(video_path="", clip=None):
    clip = core.bs.VideoSource(source=video_path)

    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

    clip = rife(clip, model="4.26.heavy", sc=True, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")

    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s="709")
    return clip
