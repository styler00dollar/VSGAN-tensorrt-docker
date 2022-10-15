import sys

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

# video imports
from src.vfi_inference import vfi_inference
from src.vfi_model import video_model

from src.rife import RIFE
from src.IFRNet import IFRNet
from src.GMFupSS import GMFupSS
from src.eisai import EISAI
from src.film import FILM
from src.M2M import M2M
from src.sepconv_enhanced import sepconv

# upscale imports
from src.upscale_inference import upscale_frame_skip
from src.pan import PAN_inference
from src.realbasicvsr import realbasicvsr_model
from src.egvsr import egvsr_model
from src.cugan import cugan_inference
from vsbasicvsrpp import BasicVSRPP
from vsswinir import SwinIR
from src.SRVGGNetCompact import SRVGGNetCompactRealESRGAN
from src.esrgan import ESRGAN_inference

# image processing imports
from src.scunet import scunet_inference

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 4  # can influence ram usage
# only needed if you are inside docker
core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")

video_path = "test.mp4"
# cfr video
clip = core.ffms2.Source(source=video_path)
# vfr video (untested)
# clip = core.ffms2.Source(source='input.mkv', fpsnum = 24000, fpsden = 1001)

# resizing with descale
# Debilinear, Debicubic, Delanczos, Despline16, Despline36, Despline64, Descale
# clip = core.descale.Debilinear(clip, 1280, 720)

###############################################
# COLORSPACE
###############################################
# convert colorspace
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
# convert colorspace + resizing
# clip = vs.core.resize.Bicubic(
#    clip, width=848, height=480, format=vs.RGBS, matrix_in_s="709"
# )

###############################################

# these demos work out of the box because docker also downloads the needed models, if you want other models, just add them
# you can combine everything however you want

######
# dedup tools
######
# from src.scene_detect import find_scenes
# skip_frame_list = find_scenes(video_path, threshold=30)

# from src.dedup import get_duplicate_frames_with_vmaf
# skip_frame_list += get_duplicate_frames_with_vmaf(video_path)

# to use for upscaling, apply this after upscaling
# clip = upscale_frame_skip(clip, skip_frame_list)

######

###############################################
# MODELS (CUDA)
###############################################
# VFI

# VFI example for jit models
# clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")

# select desired model
model_inference = RIFE(
    scale=1, fastmode=False, ensemble=True, model_version="rife46", fp16=False
)
# model_inference = IFRNet(model="small", fp16=False)
# model_inference = GMFupSS()
# model_inference = EISAI() # 960x540
# model_inference = FILM(model_choise="vgg")
# model_inference = M2M() # only 2x supported, will always be 2x
# model_inference = sepconv() # only 2x supported, will always be 2x
clip = vfi_inference(
    model_inference=model_inference, clip=clip, skip_frame_list=[], multi=2
)

######
# if you want to use dedup or scene change detect for external vs plugins like mlrt, use vfi_frame_merger

# workaround to use mlrt for video interpolation
# clip1 = core.std.DeleteFrames(clip, frames=0)
# clip2 = core.std.StackHorizontal([clip1, clip])
# clip2 = core.trt.Model(
#    clip2,
#    engine_path="/workspace/tensorrt/rife46_onnx16_1080_2input.engine",
#    num_streams=6,
# )
# clip2=core.std.Crop(clip2,right=1920)
# clip1 = core.std.Interleave([clip, clip])
# clip2 = core.std.Interleave([clip, clip2])

# skipping all duplicated / scene change frames
# clip = vfi_frame_merger(clip1, clip2, skip_frame_list)

######
# UPSCALING

# vs-mlrt (you need to create the engine yourself)
# clip = core.trt.Model(
#    clip,
#    engine_path="/workspace/tensorrt/real2x.engine",
#    tilesize=[854, 480],
#    num_streams=6,
# )
# vs-mlrt (DPIR)
# DPIR does need an extra channel
# strength = 10.0
# noise_level = clip.std.BlankClip(format=vs.GRAYS, color=strength / 100)
# clip = core.trt.Model(
#    [clip, noise_level],
#    engine_path="dpir.engine",
#    tilesize=[1280, 720],
#    num_streams=2,
# )

# SwinIR
# clip = SwinIR(clip, task="lightweight_sr", scale=2)

# ESRGAN / RealESRGAN
# tta_mode 1-7, means the amount of times the image gets processed while being mirrored
# clip = ESRGAN_inference(
#    clip=clip,
#    model_path="/workspace/RealESRGAN_x4plus_anime_6B.pth",
#    tile_x=480,
#    tile_y=480,
#    tile_pad=16,
#    fp16=False,
#    tta=False,
#    tta_mode=1,
# )

# RealESRGAN Anime Video example
# backends: tensorrt, cuda, onnx, quantized_onnx
# clip = SRVGGNetCompactRealESRGAN(clip, scale=2, fp16=True, backend_inference="tensorrt")

# EGVSR
# clip = egvsr_model(clip, interval=15)

# BasicVSR++
# 0 = REDS, 1 = Vimeo-90K (BI), 2 = Vimeo-90K (BD), 3 = NTIRE 2021 - Track 1, 4 = NTIRE 2021 - Track 2, 5 = NTIRE 2021 - Track 3
# clip = BasicVSRPP(
#    clip,
#    model=1,
#    interval=30,
#    tile_x=0,
#    tile_y=0,
#    tile_pad=16,
#    device_type="cuda",
#    device_index=0,
#    fp16=False,
#    cpu_cache=False,
# )

# RealBasicVSR
# clip = realbasicvsr_model(clip, interval=15, fp16=True)

# cugan
# scales: 2 | 3 | 4, kind_model: no_denoise | denoise3x | conservative, backend_inference: cuda | onnx, pro: True/False (only available for 2x and 3x scale)
# only cuda supports tiling
# clip = cugan_inference(
#    clip,
#    fp16=True,
#    scale=2,
#    kind_model="no_denoise",
#    backend_inference="cuda",
#    tile_x=512,
#    tile_y=512,
#    tile_pad=10,
#    pre_pad=0,
# )

# PAN
# scale = 2 | 3 | 4
# clip = PAN_inference(clip, scale=2, fp16=True)
# SCUNet
# clip = scunet_inference(
#    clip = clip,
#    fp16 = True,
#    tile_x = 384,
#    tile_y = 384,
#    tile_pad = 10,
#    pre_pad = 0,
# )

###############################################
# ncnn
###############################################
# from src.SRVGGNetCompact_ncnn import SRVGGNetCompactRealESRGAN_ncnn

# Rife ncnn (C++)
# https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan
# clip = core.misc.SCDetect(clip=clip, threshold=0.100)
# clip = core.rife.RIFE(
#    clip,
#    model=9,
#    multiplier=2,
#    gpu_id=0,
#    gpu_thread=4,
#    tta=False,
#    uhd=False,
#    skip=True,
#    sc=True,
# )

# compact example
# clip = SRVGGNetCompactRealESRGAN(
#    clip,
#    scale=2,
#    fp16=True,
#    backend_inference="ncnn",
#    param_path="test.param",
#    bin_path="test.bin",
# )

###############################################
# OUTPUT
###############################################
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()
