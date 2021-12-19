import sys
sys.path.append('/workspace/tensorrt/')
from src.vsgan import VSGAN
import vapoursynth as vs

tmp_dir = "tmp/"
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 16
core.std.LoadPlugin(path='/usr/lib/x86_64-linux-gnu/libffms2.so')
with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    txt = f.readlines()
clip = core.ffms2.Source(source=txt)
# convert colorspace
#clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s='709')
# convert colorspace + resizing
clip = vs.core.resize.Bicubic(clip, width=848, height=480, format=vs.RGBS, matrix_in_s='709')
# currently only taking normal esrgan models
#clip = VSGAN(clip, device="cuda").load_model_ESRGAN("4x_fatal_Anime_500000_G.pth").run(overlap=16).clip
clip = VSGAN(clip, device="cuda").load_model_RealESRGAN("RealESRGAN_x4plus_anime_6B.pth").run(overlap=16).clip
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()