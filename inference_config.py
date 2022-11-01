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
from src.IFUNet import IFUNet
from src.stmfnet import STMFNet
from src.rife_trt import rife_trt 

# upscale imports
from src.upscale_inference import upscale_frame_skip, upscale_inference
from src.pan import PAN_inference
from src.realbasicvsr import realbasicvsr_inference
from src.egvsr import egvsr_inference
from src.cugan import cugan_inference
from vsbasicvsrpp import BasicVSRPP
from vsswinir import SwinIR
from src.SRVGGNetCompact import compact_inference
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


def inference_clip(video_path):
    # cfr video
    clip = core.ffms2.Source(source=video_path)
    # vfr video (untested)
    # clip = core.ffms2.Source(source='input.mkv', fpsnum = 24000, fpsden = 1001)

    # resizing with descale
    # Debilinear, Debicubic, Delanczos, Despline16, Despline36, Despline64, Descale
    # clip = core.descale.Debilinear(clip, 1280, 720)

    ###############################################
    # SIMILARITY
    # Set properties in clip for it to be applied
    
    # SSIM for deduplication in frame interpolation
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    # 0 = PSNR, 1 = PSNR-HVS, 2 = SSIM, 3 = MS-SSIM, 4 = CIEDE2000
    clip = core.vmaf.Metric(clip, offs1, 2)

    # SCENE DETECT
    clip = core.misc.SCDetect(clip=clip, threshold=0.100)
    ###############################################
    # COLORSPACE
    ###############################################
    
    # convert colorspace
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
    # convert colorspace + resizing
    #clip = vs.core.resize.Bicubic(
    #    clip, width=1280, height=720, format=vs.RGBS, matrix_in_s="709"
    #)

    ###############################################
    # MODELS
    ###############################################
    ######
    # VFI
    ######

    # VFI example for jit models
    # clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")

    # Rife: model "rife40" up to "rife46" and "sudo_rife4"
    model_inference = RIFE(
        scale=1, fastmode=False, ensemble=True, model_version="rife46", fp16=False
    )

    # IFRNet: model="small" or "large"
    # model_inference = IFRNet(model="small", fp16=False)

    # model_inference = GMFupSS(partial_fp16=False)

    # model_inference = EISAI() # 960x540

    # FILM: model_choise="style", "l1" or "vgg"
    # model_inference = FILM(model_choise="vgg")

    # model_inference = M2M()

    # model_inference = sepconv() # only 2x supported because architecture only outputs one image

    # model_inference = IFUNet()

    #model_inference = STMFNet()  # only 2x supported because architecture only outputs one image

    clip = vfi_inference(
        model_inference=model_inference, clip=clip, multi=2
    )

    #clip = rife_trt(clip, multi = 2, scale = 1.0, device_id = 0, num_streams = 2, engine_path = "/workspace/tensorrt/rife46.engine")


    ######
    # UPSCALING WITH TENSORRT
    ######
    
    # vs-mlrt (you need to create the engine yourself, read the readme)
    # clip = core.trt.Model(
    #    clip,
    #    engine_path="/workspace/tensorrt/real2x.engine",
    #    tilesize=[854, 480],
    #    overlap=[0 ,0],
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

    ######
    # CUDA (upscaling/denoising)
    # if possible, use mlrt from above instead due to speed
    ######
    
    # upscale_model_inference = PAN_inference(scale = 2, fp16 = True)
    
    # upscale_model_inference = egvsr_inference(scale=4)
    
    # CUGAN: kind_model="no_denoise", "conservative" or "denoise3x"
    # upscale_model_inference = cugan_inference(fp16=True,scale=2,kind_model="no_denoise")
    
    # upscale_model_inference = scunet_inference(fp16 = True)
    
    # ESRGAN: tta is in the range between 1 and 7
    # upscale_model_inference = ESRGAN_inference(model_path="/workspace/tensorrt/models/RealESRGAN_x4plus_anime_6B.pth", fp16=False, tta=False, tta_mode=1)
    
    # Compact: no tiling allowed due to onnx-tensorrt not allowing dynamic shapes, use mlrt instead though
    # upscale_model_inference = compact_inference(scale=2, fp16=True, clip=clip)
    
    # upscale_model_inference = realbasicvsr_inference(fp16=True)
    
    # clip = upscale_inference(upscale_model_inference, clip, tile_x=512, tile_y=512, tile_pad=10, pre_pad=0)

    ######
    # external vs plugins
    ######
    
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

    # SwinIR
    # clip = SwinIR(clip, task="lightweight_sr", scale=2)

    ###############################################
    # ncnn (works in docker, but only on linux, because wsl on windows does not support vulkan)
    ###############################################
    
    # Rife ncnn (C++)
    # Model list can be found in https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan
    # clip = core.misc.SCDetect(clip=clip, threshold=0.100)
    # clip = core.rife.RIFE(
    #    clip,
    #    model=9,
    #    factor_num=2,
    #    gpu_id=0,
    #    gpu_thread=4,
    #    tta=False,
    #    uhd=False,
    #    skip=True,
    #    sc=True,
    # )

    ###############################################
    # exotic
    ###############################################
    
    # if you want to use dedup or scene change detect for external vs plugins like mlrt, use vfi_frame_merger
    # example for rife with TensorRT 8.5, but that isn't in the docker yet due to nvidia not publishing the install files and compatibility reasons
    
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
    # clip = vfi_frame_merger(clip1, clip2)
    
    ###############################################
    # OUTPUT
    ###############################################
    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
