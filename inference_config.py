import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

# video imports
from src.vfi_inference import vfi_inference, vfi_frame_merger
from src.vfi_model import video_model

from src.rife import RIFE
from src.GMFupSS import GMFupSS
from src.GMFSS_union import GMFSS_union
from src.rife_trt import rife_trt
from src.cain_trt import cain_trt
from src.GMFSS_Fortuna_union import GMFSS_Fortuna_union
from src.GMFSS_Fortuna import GMFSS_Fortuna
from src.scene_detect import scene_detect

from src.color_transfer import vs_color_match

core = vs.core
core.num_threads = 4  # can influence ram usage
# only needed if you are inside docker
core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")
core.std.LoadPlugin(path="/usr/local/lib/x86_64-linux-gnu/libawarpsharp2.so")


def inference_clip(video_path="", clip=None):
    # ddfi is passing clip
    if clip is None:
        # cfr video
        # clip = core.ffms2.Source(source=video_path, cache=False)
        # vfr video
        # clip = core.ffms2.Source(source=video_path, fpsnum = 24000, fpsden = 1001, cache=False)
        # vfr video (automatically set num and den)
        # clip = core.ffms2.Source(source=video_path, fpsnum = -1, fpsden = 1, cache=False)

        # lsmash
        # clip = core.lsmas.LWLibavSource(source=video_path)
        # lsmash with hw decoding preferred
        # clip = core.lsmas.LWLibavSource(source=video_path, prefer_hw=3)
        
        # bestsource
        clip = core.bs.VideoSource(source=video_path)

    ###############################################
    # RESIZE
    ###############################################
    # resizing with descale
    # Debilinear, Debicubic, Delanczos, Despline16, Despline36, Despline64, Descale
    # clip = core.descale.Debilinear(clip, 1280, 720)

    ###############################################
    # SIMILARITY
    # Set properties in clip for it to be applied
    # SSIM for deduplication in frame interpolation

    # offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    # offs1 = core.std.CopyFrameProps(offs1, clip)
    # 0 = PSNR, 1 = PSNR-HVS, 2 = SSIM, 3 = MS-SSIM, 4 = CIEDE2000
    # clip = core.vmaf.Metric(clip, offs1, 2)

    # SCENE DETECT
    # clip = core.misc.SCDetect(clip=clip, threshold=0.100)
    # model based scene detect
    # clip = scene_detect(clip, model_name="efficientnetv2_b0", thresh=0.98, fp16=False)

    # apply model based scene detect
    # clip_orig = vs.core.std.Interleave([clip_orig] * 2)
    # clip = vfi_frame_merger(clip_orig, clip)

    # DEDUP (requires you to call "vspipe parse.py -p ." to generate infos_running.txt and tmp.txt)
    # from src.dedup import get_dedup_frames
    # frames_duplicated, frames_duplicating = get_dedup_frames()
    # clip = core.std.DeleteFrames(clip, frames_duplicated)
    # do upscaling here
    # clip = core.std.DuplicateFrames(clip, frames_duplicating)
    ###############################################
    # COLORSPACE
    ###############################################

    # convert colorspace
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
    # clip = vs.core.resize.Spline64(clip, format=vs.RGBS, matrix_in_s="709", transfer_in_s="linear")

    # convert colorspace + resizing
    # clip = vs.core.resize.Bicubic(
    #    clip, width=1280, height=720, format=vs.RGBS, matrix_in_s="709"
    # )

    ###############################################
    # MODELS
    ###############################################
    # in rare cases it can happen that image range is not 0-1 and that resulting in big visual problems, clamp input
    # clip = core.akarin.Expr(clip, "x 0 1 clamp")
    # clip = clip.std.Expr("x 0 max 1 min")
    # clip = core.std.Limiter(clip, max=1, planes=[0,1,2])

    ######
    # VFI
    ######

    # VFI example for jit models
    # clip = video_model(clip, fp16=False, model_path="/workspace/rvpV1_105661_G.pt")

    # Rife: model "rife40" up to "rife410" and "sudo_rife4"
    # model_inference = RIFE(
    #    scale=1, fastmode=True, ensemble=False, model_version="rife46", fp16=True
    # )

    # model_inference = GMFupSS(partial_fp16=False)
    # model_inference = GMFSS_union(partial_fp16=False)

    # model_inference = GMFSS_Fortuna_union()

    # model_inference = GMFSS_Fortuna()

    # clip = vfi_inference(
    #    model_inference=model_inference, clip=clip, multi=2, metric_thresh=0.999
    # )

    # clip = rife_trt(clip, multi = 2, scale = 1.0, device_id = 0, num_streams = 2, engine_path = "/workspace/tensorrt/rife46.engine")

    # clip = cain_trt(clip, device_id = 0, num_streams = 4, engine_path = "/workspace/tensorrt/rvp.engine")

    ######
    # UPSCALING WITH TENSORRT
    ######
    # vs-mlrt (you need to create the engine yourself, read the readme)
    clip = core.trt.Model(
        clip,
        engine_path="/workspace/tensorrt/cugan.engine",
        # tilesize=[854, 480],
        overlap=[0, 0],
        num_streams=4,
    )

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
    # DDFI
    # you need to use 8x interp for this
    ######
    # advanced example with pytorch vfi + dedup + scene change + upscaling

    # offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    # offs1 = core.std.CopyFrameProps(offs1, clip)
    # clip = core.vmaf.Metric(clip, offs1, 2)
    # clip = core.resize.Bicubic(clip, width=1280, height=720, format=vs.RGBS, matrix_in=1)

    # clip = core.misc.SCDetect(clip=clip, threshold=0.100)

    # model_inference = GMFupSS(partial_fp16=True)
    # clip = vfi_inference(
    #     model_inference=model_inference, clip=clip, multi=8, metric_thresh=0.999
    # )

    # clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    # offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    # offs1 = core.std.CopyFrameProps(offs1, clip)
    # clip = core.vmaf.Metric(clip, offs1, 2)
    # clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")

    # clip = core.trt.Model(
    #     clip,
    #     engine_path="/content/model.engine",
    #     num_streams=3,
    # )

    ####
    # Color Transfer
    ####

    # original_clip = clip
    # original_clip = original_clip.resize.Spline16(format=vs.RGB24, matrix_in_s="470bg")
    # clip = clip.resize.Spline16(format=vs.RGB24, matrix_in_s="470bg")
    # clip = vs_color_match(clip, original_clip, method="mkl")

    ###
    # Other
    ###
    # does not accept rgb clip, convert to yuv first
    # clip = core.warp.AWarpSharp2(clip, thresh=128, blur=2, type=0, depth=[16, 8, 8], chroma=0, opt=True, planes=[0,1,2], cplace="mpeg1")

    # more information here: https://github.com/HolyWu/vs-dpir/blob/master/vsdpir/__init__.py
    # clip = dpir(clip, num_streams = 4, nvfuser = False, cuda_graphs = False, trt = True, trt_cache_path = "/workspace/tensorrt/", task = "deblock", strength = 50, tile_w = 0, tile_h = 0, tile_pad= 8)

    # clip = core.cas.CAS(clip, sharpness=0.5)

    ###############################################
    # OUTPUT
    ###############################################
    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
