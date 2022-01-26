import sys
sys.path.append('/workspace/tensorrt/')
import vapoursynth as vs
from src.esrgan import ESRGAN_inference # esrgan and realesrgan
from src.SRVGGNetCompact import SRVGGNetCompactRealESRGAN # realesrgan anime video
from src.vfi_model import video_model # any vfi model, in this case rvp1 as demonstration
from src.sepconv_enhanced import sepconv_model # uses cupy, no tensorrt
from src.rife import RIFE # tensorrt not possible
from vsswinir import SwinIR # https://github.com/HolyWu/vs-swinir # currently not tensorrt, didn't try
from src.egvsr import egvsr_model # currently not tensorrt
from vsbasicvsrpp import BasicVSRPP
from src.realbasicvsr import realbasicvsr_model

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 16 # can influence ram usage
# only needed if you are inside docker
core.std.LoadPlugin(path='/usr/lib/x86_64-linux-gnu/libffms2.so')

# cfr video
clip = core.ffms2.Source(source='test.mkv')
# vfr video (untested)
#clip = core.ffms2.Source(source='input.mkv', fpsnum = 24000, fpsden = 1001)
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
# MODELS (CUDA)
###############################################
# sepconv
#clip = sepconv_model(clip)
# RIFE4
#clip = RIFE(clip, multi = 2, scale = 1.0, fp16 = True, fastmode = False, ensemble = True)
# VFI example for jit models
#clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")
# SwinIR
#clip = SwinIR(clip, task="lightweight_sr", scale=2)
# ESRGAN / RealESRGAN
#clip = ESRGAN_inference(clip=clip, model_path="/workspace/4x_fatal_Anime_500000_G.pth", tile_x=400, tile_y=400, tile_pad=10, fp16=False)
#clip = ESRGAN_inference(clip=clip, model_path="/workspace/RealESRGAN_x4plus_anime_6B.pth", tile_x=480, tile_y=480, tile_pad=16, fp16=False)
# RealESRGAN Anime Video example
#clip = SRVGGNetCompactRealESRGAN(clip, scale=2, fp16=True)
# EGVSR
clip = egvsr_model(clip, interval=15)
# BasicVSR++
# 0 = REDS, 1 = Vimeo-90K (BI), 2 = Vimeo-90K (BD), 3 = NTIRE 2021 - Track 1, 4 = NTIRE 2021 - Track 2, 5 = NTIRE 2021 - Track 3
#clip = BasicVSRPP(clip, model = 1, interval = 30, tile_x = 0, tile_y = 0, tile_pad = 16, device_type = 'cuda', device_index = 0, fp16 = False, cpu_cache = False)
# RealBasicVSR
#clip = realbasicvsr_model(clip, interval=15, fp16=True)

###############################################
# [NOT IN DOCKER] MODELS (NCNN)
# Only recommended for AMD GPUS, further instructions in README
###############################################
#from src.SRVGGNetCompact_ncnn import SRVGGNetCompactRealESRGAN_ncnn

# Rife ncnn
# 0 = rife-v3.1, 1 = rife-v3.0, 2 = rife-v2.4, 3 = rife-v2, 4 = rife-anime
#clip = core.rife.RIFE(clip, model=0, gpu_id=0, gpu_thread=2, tta=False, uhd=False, sc=False, list_gpu=False)
# RealESRGAN example
#clip = SRVGGNetCompactRealESRGAN_ncnn(clip, gpuid=0, model="models-DF2K", tta_mode=False, scale = 2, tilesize=0, param_path = None, bin_path = None)
# Waifu2x
# 0 = upconv_7_anime_style_art_rgb, 1 = upconv_7_photo, 2 = cunet (For 2D artwork. Slow, but better quality.)
#clip = core.w2xnvk.Waifu2x(clip, noise=0, scale=2, model=0, tile_size=0, gpu_id=0, gpu_thread=0, precision=16)

###############################################
# OUTPUT
###############################################
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
