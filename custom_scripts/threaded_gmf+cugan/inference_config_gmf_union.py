import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

from src.vfi_inference import vfi_inference
from src.GMFupSS import GMFupSS
from src.GMFSS_union import GMFSS_union
from src.scene_detect import scene_detect

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 4  # can influence ram usage
core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")

clip = core.ffms2.Source(
    source=globals()["source"],
    width=1280,
    height=720,
    fpsnum=24000,
    fpsden=1001,
    cache=False,
)
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

clip = scene_detect(clip, model_name="efficientnetv2b0+rife46", thresh=0.98)
# clip = scene_detect(clip, model_name="efficientnetv2_b0", thresh=0.98)
# model_inference = GMFupSS(partial_fp16=True)
model_inference = GMFSS_union(partial_fp16=True)
clip = vfi_inference(
    model_inference=model_inference, clip=clip, multi=2, metric_thresh=0.999
)

clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
