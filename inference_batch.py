import sys
sys.path.append('/workspace/tensorrt/')
import os
import vapoursynth as vs
from src.esrgan import ESRGAN_inference # esrgan and realesrgan
from src.SRVGGNetCompact import SRVGGNetCompactRealESRGAN # realesrgan anime video
from src.vfi_model import video_model # any vfi model, in this case rvp1 as demonstration
from src.sepconv_enhanced import sepconv_model # uses cupy, no tensorrt
from src.rife import RIFE # tensorrt not possible
from vsswinir import SwinIR # https://github.com/HolyWu/vs-swinir # currently not tensorrt, didn't try
from src.egvsr import egvsr_model # currently not tensorrt

tmp_dir = "tmp/"
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 16
core.std.LoadPlugin(path='/usr/lib/x86_64-linux-gnu/libffms2.so')
with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    txt = f.readlines()

# cfr video
clip = core.ffms2.Source(source=txt)
# vfr video (untested)
#clip = core.ffms2.Source(source=txt, fpsnum = 24000, fpsden = 1001)
###############################################
# COLORSPACE
###############################################
# convert colorspace
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s='709')
# convert colorspace + resizing
#clip = vs.core.resize.Bicubic(clip, width=848, height=480, format=vs.RGBS, matrix_in_s='709')

###############################################

# these demos work out of the box because docker also downloads the needed models, if you want other models, just add them
# you can combine everything however you want

###############################################
# MODELS
###############################################
# sepconv
#clip = sepconv_model(clip)
# RIFE4
#clip = RIFE(clip, multi = 2, scale = 1.0, fp16 = True)
# VFI example for jit models
#clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")
# SwinIR
#clip = SwinIR(clip, task="lightweight_sr", scale=2)
# ESRGAN / RealESRGAN
clip = ESRGAN_inference(clip=clip, model_path="/workspace/4x_fatal_Anime_500000_G.pth", tile_x=400, tile_y=400, tile_pad=10, fp16=False)
#clip = ESRGAN_inference(clip=clip, model_path="/workspace/RealESRGAN_x4plus_anime_6B.pth", tile_x=400, tile_y=400, tile_pad=10, fp16=False)
# RealESRGAN Anime Video example
clip = SRVGGNetCompactRealESRGAN(clip, scale=2, fp16=True)
# EGVSR
#clip = egvsr_model(clip)

###############################################
# OUTPUT
###############################################
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()