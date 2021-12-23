import sys
sys.path.append('/workspace/tensorrt/')
import os
import vapoursynth as vs
from src.vsgan import VSGAN # esrgan and realesrgan
from src.SRVGGNetCompact import SRVGGNetCompactRealESRGAN # realesrgan anime video
from src.vfi_model import video_model # any vfi model, in this case rvp1 as demonstration
from src.sepconv_enhanced import sepconv_model # uses cupy, no tensorrt
from vsrife import RIFE # https://github.com/HolyWu/vs-rife/ # tensorrt not possible
from vsswinir import SwinIR # https://github.com/HolyWu/vs-swinir # currently not tensorrt, didn't try

tmp_dir = "tmp/"
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 16
core.std.LoadPlugin(path='/usr/lib/x86_64-linux-gnu/libffms2.so')
with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    txt = f.readlines()
clip = core.ffms2.Source(source=txt)
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
#clip = RIFE(clip)
# VFI example for jit models
#clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")
# SwinIR
#clip = SwinIR(clip, task="lightweight_sr", scale=2)
# ESRGAN example (also has tiling)
clip = VSGAN(clip, device="cuda", fp16=False).load_model_ESRGAN("/workspace/4x_fatal_Anime_500000_G.pth").run(overlap=16).clip
# RealESRGAN example (also has tiling)
#clip = VSGAN(clip, device="cuda", fp16=False).load_model_RealESRGAN("/workspace/RealESRGAN_x4plus_anime_6B.pth").run(overlap=16).clip
# RealESRGAN Anime Video example
#clip = SRVGGNetCompactRealESRGAN(clip, scale=2, fp16=True)
# EGVSR
#clip = egsvr_model(clip)

###############################################
# OUTPUT
###############################################
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()