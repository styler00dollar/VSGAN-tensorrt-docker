# VSGAN-tensorrt-docker

Repository to use super resolution models and video frame interpolation models and also trying to speed them up with TensorRT. This repository contains the fastest inference code that you can find, at least I am trying to archive that. Not all codes can use TensorRT due to various reasons, but I try to add that if it works. Further model architectures are planned to be added later on.

I also created a Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/styler00dollar/VSGAN-tensorrt-docker/blob/main/Colab-VSGAN.ipynb)

Table of contents
=================

<!--ts-->
   * [Usage](#usage)
   * [Usage example](#usage-example)
   * [Video guide (depricated)](#video-guide)
   * [Deduplicated inference](#deduplicated)
   * [Scene change detection](#scene-change)
   * [vs-mlrt (C++ TRT)](#vs-mlrt)
       * [multi-gpu](#multi-gpu)
   * [ddfi](#ddfi)
   * [VFR (variable refresh rate)](#vfr)
   * [mpv](#mpv)
   * [Color transfer](#color)
   * [Benchmarks](#benchmarks)
   * [License](#license)
<!--te-->

-------

Currently working networks:
- ESRGAN with [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN) and [HolyWu/vs-realesrgan](https://github.com/HolyWu/vs-realesrgan)
- RealESRGAN / RealESERGANVideo with [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN)
- RealESRGAN ncnn with [styler00dollar/realsr-ncnn-vulkan-python](https://github.com/styler00dollar/realsr-ncnn-vulkan-python) and [media2x/realsr-ncnn-vulkan-python](https://github.com/media2x/realsr-ncnn-vulkan-python)
- [Rife4 with HolyWu/vs-rife](https://github.com/HolyWu/vs-rife/) and [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) ([rife4.0](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view) [rife4.1](https://drive.google.com/file/d/1CPJOzo2CHr8AN3GQCGKOKMVXIdt1RBR1/view) [rife4.2](https://drive.google.com/file/d/1JpDAJPrtRJcrOZMMlvEJJ8MUanAkA-99/view)
[rife4.3](https://drive.google.com/file/d/1xrNofTGMHdt9sQv7-EOG0EChl8hZW_cU/view) [rife4.4](https://drive.google.com/file/d/1eI24Kou0FUdlHLkwXfk-_xiZqKaZZFZX/view) [rife4.5](https://drive.google.com/file/d/17Bl_IhTBexogI9BV817kTjf7eTuJEDc0/view) [rife4.6](https://drive.google.com/file/d/1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE/view) [rife4.7.1](https://drive.google.com/file/d/1s2zMMIJrUAFLexktm1rWNhlIyOYJ3_ju/view) [rife4.8.1](https://drive.google.com/file/d/1wZa3SyegLPUwBQWmoDLM0MumWd2-ii63/view)
[rife4.9.2](https://drive.google.com/file/d/1UssCvbL8N-ty0xIKM5G5ZTEgp9o4w3hp/view) [rife4.10.1](https://drive.google.com/file/d/1WNot1qYBt05LUyY1O9Uwwv5_K8U6t8_x/view) [rife4.11.1](https://drive.google.com/file/d/1Dwbp4qAeDVONPz2a10aC2a7-awD6TZvL/view) [rife4.12.2](https://drive.google.com/file/d/1ZHrOBL217ItwdpUBcBtRE3XBD-yy-g2S/view))

- RIFE ncnn with [styler00dollar/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan) and [HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan)
- [SwinIR with HolyWu/vs-swinir](https://github.com/HolyWu/vs-swinir)
- [Sepconv (enhanced) with sniklaus/revisiting-sepconv](https://github.com/sniklaus/revisiting-sepconv/)
- EGVSR with [Thmen/EGVSR](https://github.com/Thmen/EGVSR) and [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- BasicVSR++ with [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- RealBasicVSR with [ckkelvinchan/RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR)
- RealCUGAN with [bilibili/ailab](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)
- PAN with [zhaohengyuan1/PAN](https://github.com/zhaohengyuan1/PAN)
- IFRNet with [ltkong218/IFRNet](https://github.com/ltkong218/IFRNet)
- M2M with [feinanshan/M2M_VFI](https://github.com/feinanshan/M2M_VFI)
- IFUNet with [98mxr/IFUNet](https://github.com/98mxr/IFUNet/)
- SCUNet with [cszn/SCUNet](https://github.com/cszn/SCUNet)
- GMFupSS with [98mxr/GMFupSS](https://github.com/98mxr/GMFupSS)
- ST-MFNet with [danielism97/ST-MFNet](https://github.com/danielism97/ST-MFNet)
- VapSR with [zhoumumu/VapSR](https://github.com/zhoumumu/VapSR)
- GMFSS_union with [HolyWu version](https://github.com/HolyWu/vs-gmfss_union), [styler00dollar/vs-gmfss_union](https://github.com/styler00dollar/vs-gmfss_union), [98mxr/GMFSS_union](https://github.com/98mxr/GMFSS_union)
- AI scene detection with [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [snap-research/EfficientFormer (EfficientFormerV2)](https://github.com/snap-research/EfficientFormer), [lucidrains/TimeSformer-pytorch](https://github.com/lucidrains/TimeSformer-pytorch) and [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)
- GMFSS_Fortuna and GMFSS_Fortuna_union with [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna), [HolyWu/vs-gmfss_fortuna](https://github.com/HolyWu/vs-gmfss_fortuna) and [styler00dollar/vs-gmfss_fortuna](https://github.com/styler00dollar/vs-gmfss_fortuna)

Also used:
- TensorRT C++ inference and python script usage with [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- ddfi with [Mr-Z-2697/ddfi-rife](https://github.com/Mr-Z-2697/ddfi-rife) (auto dedup-duplication, not an arch)
- nix with [lucasew/nix-on-colab](https://github.com/lucasew/nix-on-colab)
- custom ffmpeg with [styler00dollar/ffmpeg-static-arch-docker](https://github.com/styler00dollar/ffmpeg-static-arch-docker)
- lsmash with [AkarinVS/L-SMASH-Works](https://github.com/AkarinVS/L-SMASH-Works)
- wwxd with [dubhater/vapoursynth-wwxd](https://github.com/dubhater/vapoursynth-wwxd)
- scxvid with [dubhater/vapoursynth-scxvid](https://github.com/dubhater/vapoursynth-scxvid)
- trt precision check and upscale frame skip with [mafiosnik777/enhancr](https://github.com/mafiosnik777/enhancr)

Model | ESRGAN | SRVGGNetCompact | Rife | SwinIR | Sepconv | EGVSR | BasicVSR++ | Waifu2x | RealBasicVSR | RealCUGAN | DPIR | PAN | IFRNet | M2M | IFUNet | SCUNet | GMFupSS | ST-MFNet | VapSR | GMFSS_union | GMFSS_Fortuna / GMFSS_Fortuna_union
---  | ------- | --------------- | ---- | ------ | ------- | ----- | ---------- | ------- | ------------ | --------- | ---- | ---- | --- | ------ | --- | ------ | ----- | ------ | ---- | ---- | --- 
CUDA | - | - | yes (4.0-4.12) | [yes](https://github.com/HolyWu/vs-swinir/tree/master/vsswinir) | [yes](http://content.sniklaus.com/resepconv/network-paper.pytorch) | [yes](https://github.com/Thmen/EGVSR/raw/master/pretrained_models/EGVSR_iter420000.pth) | [yes](https://github.com/HolyWu/vs-basicvsrpp/releases/tag/model) | - | [yes](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) | [yes](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD) | - | [yes](https://github.com/zhaohengyuan1/PAN/tree/master/experiments/pretrained_models) | [yes](https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=0) | [yes](https://drive.google.com/file/d/1dO-ArTLJ4cMZuN6dttIFFMLtp4I2LnSG/view) | [yes](https://drive.google.com/file/d/1psrM4PkPhuM2iCwwVngT0NCtx6xyiqXa/view) | [yes](https://github.com/cszn/SCUNet/blob/main/main_download_pretrained_models.py) | [yes](https://github.com/98mxr/GMFupSS/tree/main/train_log) | [yes](https://drive.google.com/file/d/1s5JJdt5X69AO2E2uuaes17aPwlWIQagG/view) | - | yes ([vanilla](https://drive.google.com/file/d/1AsA7a4HNR4RjCeEmNUJWy5kY3dBC-mru/view) / [wgan](https://drive.google.com/file/d/1GAp9DljP1RCQXz0uu_GNn751NBMEQOUB/view)) | [base](https://drive.google.com/file/d/1BKz8UDAPEt713IVUSZSpzpfz_Fi2Tfd_/view) / [union](https://drive.google.com/file/d/1Mvd1GxkWf-DpfE9OPOtqRM9KNk20kLP3/view)
TensorRT | yes (torch_tensorrt / C++ TRT) | yes (onnx_tensorrt / C++ TRT) [v2](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/RealESRGANv2_v1.7z), [v3](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/RealESRGANv3_v1.7z) | yes | - | - | - | - | [yes (C++ TRT)](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/waifu2x_v3.7z) | - | [yes (C++ TRT)](https://github.com/AmusementClub/vs-mlrt/releases/download/v9.2/models.v9.2.7z) | [yes (C++ TRT)](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/dpir_v3.7z) | - | - | - | - | - | - | - | [yes (C++ TRT)](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models) | - | -
ncnn | - | yes, but compile yourself ([2x](https://files.catbox.moe/u62vpw.tar)) | [yes (4.0-4.12)](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/tree/master/models) | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | -

Some important things:
- `ncnn` does not work in wsl and that means it doesn't work in Windows currently. `ncnn` will only work if you use docker in linux.
- If you are on Windows, install all the latest updates first, otherwise wsl won't work properly. 21H2 minimum.
- Do not use `webm` video, webm is often broken. It can work, but don't complain about broken output afterwards. I would suggest to render webm into mp4 or mkv.
- Only use ffmpeg to determine if video is variable framerate (vfr) or not. Other programs do not seem reliable.
- Processing vfr video is dangerous, but you can try to use fpsnum and fpsden. Either use these params or render the input video into constant framerate (crf).
- `x264` can be faster than `ffmpeg`.
- The C++ VS rife extention can be faster than CUDA.
- Colabs have a weak cpu, you should try `x264` with `--opencl`. (A100 does not support NVENC and such)

<div id='usage'/>

## Usage
Get CUDA12.1 and latest Nvidia drivers. After that, follow the following steps:

**WARNING FOR WINDOWS USERS: Docker Desktop `4.17.1` is broken. I confirmed that [4.25.0](https://desktop.docker.com/win/main/amd64/126437/Docker%20Desktop%20Installer.exe) should work. Older tested versions are [4.16.3](https://desktop.docker.com/win/main/amd64/96739/Docker%20Desktop%20Installer.exe) or [4.17.0](https://desktop.docker.com/win/main/amd64/99724/Docker%20Desktop%20Installer.exe). I would recommend to use `4.25.0`. `4.17.1` results in Docker not starting which is mentioned in [this issue](https://github.com/styler00dollar/VSGAN-tensorrt-docker/issues/34).**

**ANOTHER WARNING FOR PEOPLE WITHOUT `AVX512`: Instead of using `styler00dollar/vsgan_tensorrt:latest`, which I build with my 7950x and thus with all AVX, use `chin39/vsgan_tensorrt:latest` in `compose.yaml` and the git branch `no_avx` to avoid `Illegal instruction (core dumped)` which is mentioned in [this issue](https://github.com/styler00dollar/VSGAN-tensorrt-docker/issues/48).**

Quickstart:
```bash
# if you have Windows, install Docker Desktop https://www.docker.com/products/docker-desktop/
# if you encounter issues, install one of the following versions:
# 4.16.3: https://desktop.docker.com/win/main/amd64/96739/Docker%20Desktop%20Installer.exe
# 4.17.0: https://desktop.docker.com/win/main/amd64/99724/Docker%20Desktop%20Installer.exe

# if you have Arch, install the following dependencies
yay -S docker nvidia-docker nvidia-container-toolkit docker-compose docker-buildx

# run the docker with docker-compose
# you need to be inside the vsgan folder with cli before running the following step, git clone repo and cd into it
# go into the vsgan folder, inside that folder should be compose.yaml, run this command
# you can adjust folder mounts in the yaml file
docker-compose run --rm vsgan_tensorrt
```
There are now multiple containers to choose from, if you don't want the default, then edit `compose.yaml`
and set a different tag `image: styler00dollar/vsgan_tensorrt:x` prior to running `docker-compose run --rm vsgan_tensorrt`.
- `latest`: Default docker with everything.
- The other latest image is by [chinrw](https://hub.docker.com/r/chin39/vsgan_tensorrt/tags) and built on a i9-13900K. Since The docker is built with all compatible instructions, it just crashes on cpus that
  are not compatible. Use this instead if your cpu does not support all instruction sets. Use the branch `no_avx` for that, this was the git status when the docker image was created.
- `minimal`: Bare minimum to run `ffmpeg`, `mlrt` and `lsmash`.
- `deprecated`: Container before changing dockerfile to copy stage, has same functionality as latest, but is way bigger in size. (not recommended)
- `ffmpeg_trt`: Experimental ffmpeg trt plugin without vapoursynth, only for sm_89 for now, or recompile with your own gpu compute version.
   The ffmpeg in this docker is also barebones for now. since the plugin is currently only compatible with ffmpeg4 and is not compiled with many dependencies.
   That means no av1 gpu encoding and not a lot of encoding/decoding options, but a ffmpeg trt plugin should avoid any upscaling bottleneck. 
   With this plugin you can direclty encode data that is located in the gpu without needing to copy back the data to the cpu with nvenc.

| docker image  | compressed download | extracted container | short description |
| ------------- | ------------------- | ------------------- | ----------------- |
| styler00dollar/vsgan_tensorrt:latest | 9gb | 17gb | default latest
| chin39/vsgan_tensorrt:latest | 9gb | 17gb | default latest without AVX512
| styler00dollar/vsgan_tensorrt:minimal | 4gb | 8gb | ffmpeg + mlrt + lsmash
| styler00dollar/vsgan_tensorrt:deprecated | 23gb | 43gb | old default
| styler00dollar/vsgan_tensorrt:ffmpeg_trt | 9gb | 20gb | ffmpeg c++ trt inference plugin to use trt engines with ffmpeg directly without vapoursynth

Piping usage:
```
# you can use it in various ways, ffmpeg example
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv -y
# nvencc example
vspipe -c y4m inference.py - | nvencc -i pipe: --codec av1 -o example.mkv
# x264 example
vspipe -c y4m inference.py - | x264 - --demuxer y4m -o example.mkv -y
# x265 example
vspipe -c y4m inference.py - | x265 - --y4m -o example.mkv -y

# example without vspipe
ffmpeg -f vapoursynth -i inference.py example.mkv -y

# example with ffmpeg trt plugin + nvenc
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_npp=1280:720,format_cuda=rgbpf32le,tensorrt=my_engine.engine,format_cuda=nv12 -c:v hevc_nvenc -preset lossless output.mkv -y
# example with ffmpeg trt plugin + hwdownload (cpu encoding)
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf format_cuda=rgbpf32le,tensorrt=my_engine.engine,format_cuda=nv12,hwdownload,format=nv12 -vcodec ffv1 output.mkv -y
```
If docker does not want to start, try this before you use docker:
```bash
# fixing docker errors
sudo systemctl start docker
sudo chmod 666 /var/run/docker.sock
```
Linux docker autostart:
```
sudo systemctl enable --now docker
```
The following stuff is for people who want to run things from scratch.
Manual ways of downloading the docker image:
```
# Download prebuild image from dockerhub (recommended)
docker pull styler00dollar/vsgan_tensorrt:latest

# if you have `unauthorized: authentication required` problems, download the docker with 
git clone https://github.com/NotGlop/docker-drag
cd docker-drag
python docker_pull.py styler00dollar/vsgan_tensorrt:latest
docker load -i styler00dollar_vsgan_tensorrt.tar
```
Manually building docker image from scratch:
```
# Build docker manually (only required if you want to build from scratch)
# This step is not needed if you already downloaded the docker and is only needed if yo
# want to build it from scratch. Keep in mind that you need to set env variables in windows differently and
# this command will only work in linux. Run that inside that directory
DOCKER_BUILDKIT=1 docker build -t styler00dollar/vsgan_tensorrt:latest .
# If you want to rebuild from scratch or have errors, try to build without cache
DOCKER_BUILDKIT=1 docker build --no-cache -t styler00dollar/vsgan_tensorrt:latest . 
```
Manually run docker:
```
# you need to be inside the vsgan folder with cli before running the following step, git clone repo and cd into it
# the folderpath before ":" will be mounted in the path which follows afterwards
# contents of the vsgan folder should appear inside /workspace/tensorrt

docker run --privileged --gpus all -it --rm -v /home/vsgan_path/:/workspace/tensorrt styler00dollar/vsgan_tensorrt:latest

# Windows is mostly similar, but the path needs to be changed slightly:
Example for C://path
docker run --privileged --gpus all -it --rm -v /mnt/c/path:/workspace/tensorrt styler00dollar/vsgan_tensorrt:latest
docker run --privileged --gpus all -it --rm -v //c/path:/workspace/tensorrt styler00dollar/vsgan_tensorrt:latest
``` 
<div id='usage-example'/>

## Usage example

Small minimalistic example of how to configure inference. If you only want to process one video, then edit video path in `inference.py`
```
video_path = "test.mkv"
```
and then afterwards edit `inference_config.py`. Small example:

```python
import sys
sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 4
core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")

from src.rife import RIFE
from src.vfi_inference import vfi_inference

def inference_clip(video_path):
    clip = core.ffms2.Source(source=video_path, cache=False)
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
    # apply one or multiple models, will be applied in order
    model_inference = RIFE(scale=1, fastmode=False, ensemble=True, model_version="rife46", fp16=True)
    clip = vfi_inference(model_inference=model_inference, clip=clip, multi=2)
    # return clip
    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
```
Then use the commands above to render. For example:
```
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv
```

Video will be rendered without sound and other attachments. You can add that manually to the ffmpeg command.

To process videos in batch and copy their properties like audio and subtitle to another file, you need to use `main.py`. Edit filepaths and file extention:
```python
input_dir = "/workspace/tensorrt/input/"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.webm", recursive=True)
```
and configure `inference_config.py` like wanted. Afterwards just run
```
python main.py
```

<div id='video-guide'/>

## Video guide (deprecated)

**WARNING: I RECOMMEND READING THE README INSTEAD. THE VIDEO SHOULD GET RE-DONE AT SOME POINT.**

If you are confused, here is a Youtube video showing how to use Python API based TensorRT on Windows. That's the easiest way to get my code running, but I would recommend trying to create `.engine` files instead. I wrote instructions for that further down below under [vs-mlrt (C++ TRT)](#vs-mlrt). The difference in speed can be quite big. Look at [benchmarks](#benchmarks) for further details.

[![Tutorial](https://img.youtube.com/vi/B134jvhO8yk/0.jpg)](https://www.youtube.com/watch?v=B134jvhO8yk)

<div id='deduplicated'/>

## Deduplicated inference
Calculate similarity between frames with [HomeOfVapourSynthEvolution/VapourSynth-VMAF](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF).
```python
# requires yuv, convert if it isn't
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
# adding metric to clip property
# 0 = PSNR, 1 = PSNR-HVS, 2 = SSIM, 3 = MS-SSIM, 4 = CIEDE2000
offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
offs1 = core.std.CopyFrameProps(offs1, clip)
clip = core.vmaf.Metric(clip, offs1, 2)
# convert to rgbs if needed
clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
```
The properties in the clip will then be used to skip similar frames.

<div id='scene-change'/>

## Scene change detection

Scene change detection is implemented in various different ways. To use traditional scene change without ai you can do:

```python
clip = core.misc.SCDetect(clip=clip, threshold=0.100)
```
The clip property will then be used in frame interpolation inference.

Recently I started experimenting in training my own scene change detect models and I used a dataset with 272.016 images (90.884 triplets) which includes everything from animation to real video (vimeo90k + animeinterp + custom data). So these should work on any kind of video.

```python
clip = scene_detect(clip, model_name="efficientnetv2_b0", thresh=0.98)
```

**Warning: Keep in mind that different models may require a different thresh to be good.**

I think that `efficientnetv2_b0` is a good balance between speed and results. It overall did quite good. The other models which are included are not listed in an order. They looked all looked ok, but you would need to test yourself to dertermine an opinion.

My personal favorites would be `efficientnetv2_b0`, `efficientformerv2_s0`, `maxvit_small` and `swinv2_small` for video interpolation tasks. Even if they overdetect a little, the main point is to avoid bad interpolation frames and the detection of bigger differences and scene changes is key because of that. Models will have a hard time discerning if bigger differences are a scene change and handle it in their own way. Some will trigger more and some less.

Sidenote: "overdetect" is a bit hard to define with animation. There is no objective way of saying what frames are similar for drawn animation compared to irl videos. With a fast scene, fighting scene, zooming scene or scenes with particle effects covering a lot of the screen bigger differences can happen, but it does not necessarily mean a scene change. What about partial transitions and only partially changing screens? These are based on my opinion.

Model list:
- efficientnetv2_b0: Good overall
- efficientnetv2_b0+rife46
- efficientformerv2_s0: good overall
- efficientformerv2_s0+rife46
- maxvit_small: good, but can overdetect at high movement
- maxvit_small+rife46
- regnetz_005: good overall
- repvgg_b0: does barely overdetect, but seems to miss a few frames
- resnetrs50: a bit hit and miss, but does not overdetect
- resnetv2_50: might miss a bit, needs lower thresh like 0.9
- rexnet_100: not too much and not too little, not perfect tho
- swinv2_small: detects more than efficientnetv2_b0, but detects a bit too much at high movement
- swinv2_small+rife46
- TimeSformer: it's alright, but might overdetect a little

Models that I trained but seemed to be bad:
- hornet_tiny_7x7
- renset50
- STAM
- volo_d1
- tf_efficientnetv2_xl_in21k
- resnext50_32x4d
- nfnet_f0
- swsl_resnet18
- poolformer_m36
- densenet121

Interesting observations:
- Applying means/stds seemingly worsened results, despite people doing that as standard practise.
- Applying image augmentation worsened results.
- Training with higher batchsize made detections a little more stable, but maybe that was placebo and a result of more finetuning.

Comparison to traditional methods:
- [wwxd](https://github.com/dubhater/vapoursynth-wwxd) and [scxvid](https://github.com/dubhater/vapoursynth-scxvid) suffer from overdetection (at least in drawn animation).
- The json that [master-of-zen/Av1an](https://github.com/master-of-zen/Av1an) produces with `--sc-only --sc-method standard --scenes test.json` returns too little scene changes. Changing the method does not really influence a lot. Not reliable enough for vfi.
- I can't be bothered to [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) get working with vapousynth with FrameEval and by default it only works with video or image sequence as input. I may try in the future, but I don't understand why I cant just input two images.
- `misc.SCDetect` seemed like the best traditional vapoursynth method that does currently exist, but I thought I could try to improve. It struggles harder with similar colors and tends to skip more changes compared to ai methods.

<div id='vs-mlrt'/>

## vs-mlrt (C++ TRT)
You need to convert onnx models into engines. You need to do that on the same system where you want to do inference. Download onnx models from [here]( https://github.com/AmusementClub/vs-mlrt/releases/download/v7/models.v7.7z) or from [my Github page](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models). You can technically just use any ONNX model you want or convert a pth into onnx with [convert_esrgan_to_onnx.py](https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/convert_esrgan_to_onnx.py) or [convert_compact_to_onnx.py](https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/convert_compact_to_onnx.py). Inside the docker, you do one of the following commands:

Good default choice:
```
trtexec --fp16 --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference
```
With some arguments known for speedup (Assuming enough vram for 4 stream inference):
```
trtexec --fp16 --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --infStreams=4 --builderOptimizationLevel=4
```
Be aware that DPIR (color) needs 4 channels.
```
trtexec --fp16 --onnx=dpir_drunet_color.onnx --minShapes=input:1x4x8x8 --optShapes=input:1x4x720x1280 --maxShapes=input:1x4x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference
```
Rife needs 8 channels. Setting `fasterDynamicShapes0805` since trtexec recommends it.
```
trtexec --fp16 --onnx=rife.onnx --minShapes=input:1x8x64x64 --optShapes=input:1x8x720x1280 --maxShapes=input:1x8x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --preview=+fasterDynamicShapes0805
```
rvpV2 needs 6 channels, but does not support variable shapes.
```
trtexec --fp16 --onnx=rvp2.onnx --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference
```
and put that engine path into `inference_config.py`. Only do FP16 if your GPU does support it. 

Recommended arguments:
```
--tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT 
--infStreams=4 (and then using num_streams=4 in mlrt)
--builderOptimizationLevel=4 (5 can be result in segfault, default is 3)
```
Not recommended arguments which also showed reduction in speed:
```
--heuristic
--refit
--maxAuxStreams=4
--preview="+fasterDynamicShapes0805,+profileSharing0806"
--tacticSources=+CUDNN,+CUBLAS,+CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS,+JIT_CONVOLUTIONS (turning all on)
```
Testing was done on a 4090 with shuffle cugan.

**Warnings**: 
- If you use the FP16 onnx you need to use `RGBH` colorspace, if you use FP32 onnx you need to use `RGBS` colorspace in `inference_config.py` 
- Engines are system specific, don't use across multiple systems
- Don't use reuse engines for different GPUs.
- If you run out of memory, then you need to adjust the resolutions in that command. If your video is bigger than what you can input in the command, use tiling.

<div id='multi-gpu'/>

### multi-gpu

Thanks to tepete who figured it out, there is also a way to do inference on multipe GPUs.

```python
stream0 = core.std.SelectEvery(core.trt.Model(clip, engine_path="models/engines/model.engine", num_streams=2, device_id=0), cycle=3, offsets=0)
stream1 = core.std.SelectEvery(core.trt.Model(clip, engine_path="models/engines/model.engine", num_streams=2, device_id=1), cycle=3, offsets=1)
stream2 = core.std.SelectEvery(core.trt.Model(clip, engine_path="models/engines/model.engine", num_streams=2, device_id=2), cycle=3, offsets=2)
clip = core.std.Interleave([stream0, stream1, stream2])
```

<div id='ddfi'/>

## ddfi

To quickly explain what ddfi is, the repository [Mr-Z-2697/ddfi-rife](https://github.com/Mr-Z-2697/ddfi-rife) deduplicates frames and interpolates between frames. Normally, frames which are duplicated can create a stuttering visual effect and to mitigate that, a higher interpolation factor is used on scenes which have a duplicated frames to compensate. 

Visual examples from that repository:

https://user-images.githubusercontent.com/74594146/142829178-ff08b96f-9ca7-45ab-82f0-4e95be045f2d.mp4

To use it, first you need to edit `ddfi.py` to select your interpolator of choice and then also apply the desired framerate. The official code uses 8x and I suggest you do so too. Small example:
```python
clip = core.misc.SCDetect(clip=clip, threshold=0.100)
clip = core.rife.RIFE(clip, model=9, sc=True, skip=False, multiplier=8)

clip = core.vfrtocfr.VFRToCFR(
    clip, os.path.join(tmp_dir, "tsv2nX8.txt"), 192000, 1001, True
) # 23.97 * 8
``` 

Afterwards, you need to use `deduped_vfi.py` similar to how you used `main.py`. Adjust paths and file extention.

<div id='vfr'/>

## VFR
**Warning**: Using variable refresh rate video input will result in desync errors. To check if a video is do
```bash
ffmpeg -i video_Name.mp4 -vf vfrdet -f null -
```
and look at the final line. If it is not zero, then it means it is variable refresh rate. Example:
```bash
[Parsed_vfrdet_0 @ 0x56518fa3f380] VFR:0.400005 (15185/22777) min: 1801 max: 3604)
```
To go around this issue, specify `fpsnum` and `fpsden` in `inference_config.py`
```
clip = core.ffms2.Source(source='input.mkv', fpsnum = 24000, fpsden = 1001, cache=False)
```
or convert everything to constant framerate with ffmpeg.
```bash
ffmpeg -i video_input.mkv -vsync cfr -crf 10 -c:a copy video_out.mkv
```
or use my `vfr_to_cfr.py` to process a folder.

<div id='mpv'/>

## mpv
It is also possible to directly pipe the video into mpv, but you most likely wont be able to archive realtime speed. If you use a very efficient model, it may be possible on a very good GPU. Only tested in Manjaro. 
```bash
# add this to dockerfile or just execute it to install mpv
RUN apt install mpv -y && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --yes pulseaudio-utils && \
  apt-get install -y pulseaudio && apt-get install pulseaudio libpulse-dev osspd -y && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# make sure you have pulseaudio on your host system
yay -S pulseaudio

# start docker with docker-compose
# same instructions as above, but delete compose.yaml and rename compose_mpv.yaml to compose.yaml 
docker-compose run --rm vsgan_tensorrt

# start docker manually
docker run --rm -i -t \
    --network host \
    -e DISPLAY \
    -v /home/vsgan_path/:/workspace/tensorrt \
    --ipc=host \
    --privileged \
    --gpus all \
    -e PULSE_COOKIE=/run/pulse/cookie \
    -v ~/.config/pulse/cookie:/run/pulse/cookie \
    -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
    -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
    vsgan_tensorrt:latest
    
# run mpv
vspipe --y4m inference.py - | mpv -
# with custom audio and subtitles
vspipe --y4m inference.py - | mpv - --audio-file=file.aac --sub-files=file.ass
# to increase the buffer cache, you can use
--demuxer-max-bytes=250MiB
```

<div id='color'/>

## Color transfer (experimental)
A small script for color transfer is available. Currently it can only be used outside of VapourSynth. Since it uses `color-matcher` as a dependency, you need to install it first.
I only tested it on a single image for now, but it may be usable for video sequences.
```bash
pip install docutils
git clone https://github.com/hahnec/color-matcher
cd color-matcher
python setup.py install
```

You can choose between `rgb`, `lab`, `ycbcr`, `lum`, `pdf`, `sot`, `hm`, `reinhard`, `mvgd`, `mkl`, `hm-mvgd-hm` and `hm-mkl-hm`. Specify folders.
```bash
python color_transfer.py -s input -t target -o output -algo mkl -threads 8
```

<div id='benchmarks'/>

## Benchmarks

Warnings: 
- Keep in mind that these benchmarks can get outdated very fast due to rapid code development and configurations.
- The default is ffmpeg.
- ModifyFrame is depricated. Trying to use FrameEval everywhere and is used by default.
- ncnn did a lot of performance enhancements lately, so results may be a bit better.
- TensorRT docker version and ONNX opset seem to influence speed but that wasn't known for quite some time. I have a hard time pinpointing which TensorRT and ONNX opset was used. Take benchmark as a rough indicator.
- Colab may change hardware like CPU at any point.
- Sometimes it takes a very long time to reach the final speed. It can happen that not enough time was waited.
- 3090¹ (+11900k) benches most likely were affected by power lowered power limit.
- 3090² (+5950x) system provided by Piotr Rencławowicz for benchmarking purposes.
- `int8` does not automatically mean usable model. It can differ from normal inference quite a lot without adjusting the model.
- `thread_queue_size` means `-thread_queue_size 2488320`.
- "*" indicates benchmarks which were done with `vspipe file.py -p .` instead of piping into ffmpeg and rendering to avoid cpu bottleneck.

Compact (2x) | 480p | 720p | 1080p
------  | ---  | ---- | ------
rx470 vs+ncnn (np+no tile+tta off) | 2.7 | 1.6 | 0.6
1070ti vs+ncnn (np+no tile+tta off) | 4.2 | 2 | 0.9
1070ti (ONNX-TRT+FrameEval) | 12 | 6.1 | 2.8
1070ti (C++ TRT+FrameEval+num_streams=6) | 14 | 6.7 | 3
3060ti (ONNX-TRT+FrameEval) | ? | 7.1 | 3.2
3060ti (C++ TRT+FrameEval+num_streams=5) | ? | 15.97 | 7.83
3060ti VSGAN 2x | ? | 3.6 | 1.77
3060ti ncnn (Windows binary) 2x | ? | 4.2 | 1.2
3060ti Joey 2x | ? | 0.87 | 0.36
3070 (ONNX-TRT+FrameEval) | 20 | 7.55 | 3.36
3090¹ (ONNX-TRT+FrameEval) | ? | ? | 6.7
3090² (vs+TensorRT8.4+C++ TRT+vs_threads=20+num_streams=20+opset15) | 105 | 47 | 21
2x3090² (vs+TensorRT8.4+C++ TRT+num_streams=22+opset15) | 133 | 55 | 23
V100 (Colab) (vs+CUDA) | 8.4 | 3.8 | 1.6
V100 (Colab) (vs+TensorRT8+ONNX-TRT+FrameEval) | 8.3 | 3.8 | 1.7
V100 (Colab High RAM) (vs+CUDA+FrameEval) | 29 | 13 | 6
V100 (Colab High RAM) (vs+TensorRT7+ONNX-TRT+FrameEval) | 21 | 12 | 5.5
V100 (Colab High RAM) (vs+TensorRT8.2GA+ONNX-TRT+FrameEval) | 21 | 12 | 5.5
V100 (Colab High RAM) (vs+TensorRT8.4+C++ TRT+num-streams=15) | ? | ? | 6.6
A100 (Colab) (vs+CUDA+FrameEval) | 40 | 19 | 8.5
A100 (Colab) (vs+TensorRT8.2GA+ONNX-TRT+FrameEval) | 44 | 21 | 9.5
A100 (Colab) (vs+TensorRT8.2GA+C++ TRT+ffmpeg+FrameEval+num_streams=50) | 52.72 | 24.37 | 11.84
A100 (Colab) (vs+TensorRT8.2GA) (C++ TRT+x264 (--opencl)+FrameEval+num_streams=50) | 57.16 | 26.25 | 12.42
A100 (Colab) (vs+onnx+FrameEval) | 26 | 12 | 4.9
A100 (Colab) (vs+quantized onnx+FrameEval) | 26 | 12 | 5.7
A100 (Colab) (jpg+CUDA) | 28.2 (9 Threads) | 28.2 (7 Threads) | 9.96 (4 Threads)
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 79.2* | ? / 41*
6700xt (vs_threads=4+mlrt ncnn) | ? / 7.7* | ? / 3.25* | ? / 1.45*

Compact (4x) | 480p | 720p | 1080p
------  | ---  | ---- | ------
1070ti TensorRT8 docker (ONNX-TensorRT+FrameEval) | 11 | 5.6 | X
3060ti TensorRT8 docker (ONNX-TensorRT+FrameEval) | ? | 6.1 | 2.7
3060ti TensorRT8 docker 2x (C++ TRT+FrameEval+num_streams=5) | ? | 11 | 5.24
3060ti VSGAN 4x | ? | 3 | 1.3
3060ti ncnn (Windows binary) 4x | ? | 0.85 | 0.53
3060ti Joey 4x | ? | 0.25 | 0.11
A100 (Colab) (vs+CUDA+FrameEval) | 12 | 5.6 | 2.9
A100 (Colab) (jpg+CUDA) | ? | ?| 3 (4 Threads)
4090³ (TensorRT8.4GA+10 vs threads+fp16) | ? | ? / 56* (5 streams) | ? / 19.4* (2 streams)

UltraCompact (2x) | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 113.7* | ? / 52.7*
6700xt (vs_threads=4+mlrt ncnn) | ? / 14.5* | ? / 6.1* | ? / 2.76*

cugan (2x) | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
1070ti (vs+TensorRT8.4+ffmpeg+C++ TRT+num_streams=2+no tiling+opset13) | 6 | 2.7 | OOM
V100 (Colab) (vs+CUDA+ffmpeg+FrameEval) | 7 | 3.1 | ?
V100 (Colab High RAM) (vs+CUDA+ffmpeg+FrameEval) | 21 | 9.7 | 4
V100 (Colab High RAM) (vs+TensorRT8.4+ffmpeg+C++ TRT+num_streams=3+no tiling+opset13) | 30 | 14 | 6
A100 (Colab High RAM) (vs+TensorRT8.4+x264 (--opencl)+C++ TRT+vs threads=8+num_streams=8+no tiling+opset13) | 53.8 | 24.4 | 10.9
3090² (vs+TensorRT8.4+ffmpeg+C++ TRT+vs_threads=8+num_streams=5+no tiling+opset13) | 79 | 35 | 15
2x3090² (vs+TensorRT8.4+ffmpeg+C++ TRT+vs_threads=12+num_streams=5+no tiling+opset13) | 131 | 53 | 23
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 51* | ? / 22.7*
6700xt (vs_threads=4+mlrt ncnn) | ? / 3.3* | ? / 1.3* | OOM (512px tiling ? / 0.39*)

ESRGAN 4x (64mb) (23b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
1070ti TensorRT8 docker (Torch-TensorRT+ffmpeg+FrameEval) | 0.5 | 0.2 | >0.1
3060ti TensorRT8 docker (Torch-TensorRT+ffmpeg+FrameEval) | ? | 0.7 | 0.29
3060ti Cupscale (Pytorch) | ? | 0.13 | 0.044
3060ti Cupscale (ncnn) | ? | 0.1 | 0.04
3060ti Joey | ? | 0.095 | 0.043
V100 (Colab) (Torch-TensorRT8.2GA+ffmpeg+FrameEval) | 1.8 | 0.8 | ?
V100 (Colab High VRAM) (C++ TensorRT8.2GA+x264 (--opencl)+FrameEval+no tiling) | 2.46 | OOM (OpenCL) | OOM (OpenCL)
V100 (Colab High VRAM) (C++ TensorRT8.2GA+x264+FrameEval+no tiling) | 2.49 | 1.14 | 0.47
A100 (Colab) (Torch-TensorRT8.2GA+ffmpeg+FrameEval) | 5.6 | 2.6 | 1.1
3090² (C++ TRT+vs_threads=20+num_threads=2+no tiling+opset14) | 3.4 | 1.5 | 0.7
2x3090² (C++ TRT+vs_threads=20+num_threads=2+no tiling+opset14) | 7.0 | 3.2 | 1.5
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBS+op14) | ? | ? / 2.6* | ? / 1.2*

Note: The offical RealESRGAN repository uses 6b (6 blocks) for the anime model.

RealESRGAN (4x) (6b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=2) | ? | 1.7 | 0.75
V100 (Colab High RAM) (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=1+no tiling) | 6.82 | 3.15 | OOM (OpenCL) 
V100 (Colab High RAM) (vs+TensorRT8.2GA+x264+C++ TRT+num_streams=1+no tiling) | ? | ? | 1.39
A100 (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=3+no tiling) | 14.65 | 6.74 | 2.76
3090² (C++ TRT+vs_threads=20+num_threads=2+no tiling+opset14) | 11 | 4.8 | 2.3
2x3090² (C++ TRT+vs_threads=10+num_threads=2+no tiling+opset14) | 22 | 9.5 | 4.2
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 8.8* | ? / 3.9*

RealESRGAN (2x) (6b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
1070ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=1+no tiling+opset15) | 0.9 | 0.8 | 0.3
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=1) | ? | 3.12 | 1.4
V100 (Colab High RAM / 8CPU) (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=3+no tiling+opset15) | 5.09 | 4.56 | 2.02
V100 (Colab High RAM / 8CPU) (vs+TensorRT8.2GA+ffmpeg+C++ TRT+num_streams=3+no tiling+opset15) | 5.4 | 4.8 | 2.2
3090² (C++ TRT+vs_threads=20+num_threads=6+no tiling+opset16) (+dropout) | 13 | 5.8 | 2.7
2x3090² (C++ TRT+vs_threads=20+num_threads=6+no tiling+opset16) (+dropout) | 26 | 11 | 5.3
4090 (C++ TRT+TensorRT8.4GA+vs_threads=6+num_threads=6+no tiling+opset16+"--best") (+dropout) | ? | ? | ? / 12*

Rife4+vs (ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
4090 rife4.0 (fast=True) (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 415.8* | ? / 186.7*
4090 rife4.2 (fast=True) (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 418.9* | ? / 187.5*
4090 rife4.3 (fast=True) (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 419.1* | ? / 187.5*
4090 rife4.5 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 418.6* | ? / 187.6*
4090 rife4.6 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 417.8* | ? / 187*
4090 rife4.6 (ncnn+num_threads=4+num_streams=2+RGBS) | ? | ? / 139.3* | ? / 63*
Steam Deck rife4.6 (ncnn+RGBS) | ? | ? / 19.2* | ? / 8.8*
4090 rife4.7 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 278.4* | ? / 135.7*
Steam Deck rife4.7 (ncnn+RGBS) | ? | ? / 15.2* | ? / 7.2*
4090 rife4.7 (ncnn+num_threads=4+num_streams=2+RGBS) | ? | ? / 130.5* | ? / 58.2*
4090 rife4.10 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 247* | ? / 123*
4090 rife4.10 (ncnn+num_threads=4+num_streams=2+RGBS) | ? | ? / 120.7* | ? / 53.3*

Rife4+vs (ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
4090 rife4.6 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 350.7* | ? / 158.7*
4090 rife4.7 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 231.7* | ? / 104.7*
4090 rife4.10 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 206.9* | ? / 91.9*

* Benchmarks made with [HolyWu version](https://github.com/HolyWu/vs-gmfss_union) with threading and partial TensorRT and without setting `tactic` to `JIT_CONVOLUTIONS` and `EDGE_MASK_CONVOLUTIONS` due to performance penalty. I added [a modified version](https://github.com/styler00dollar/vs-gmfss_union) as a plugin to VSGAN, but I need to add enhancements to my own repo later.

GMFSS_union | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
4090 (num_threads=8, num_streams=3, RGBH, TRT8.6, matmul_precision=medium) | ? | ? / 44.6* | ? / 15.5*

GMFSS_fortuna_union | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
4090 (num_threads=8, num_streams=2, RGBH, TRT8.6.1, matmul_precision=medium) | ? | ? / 50.4* | ? / 16.9*
4090 (num_threads=8, num_streams=2, RGBH, TRT8.6.1, matmul_precision=medium, @torch.compile(mode="default", fullgraph=True)) | ? | ? / 50.6* | ? / 17*

EGVSR (4x, interval=5) | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
1070ti | 4.4 | Ram OOM / 2.2* | VRAM OOM

RealBasicVSR | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
1070ti | 0.3 | OOM | OOM
A100 (Colab) | 1.2 | ? | ?

Sepconv | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
V100 (Colab) | 22 | 11 | 4.9
3090² (vs+CUDA) | 30 | 14 | 6.2

CAIN (2 groups) | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
A100 (Colab) | 76 | 47 | 25
3090² (vs+CUDA) | 120 | 65 | 31

FILM | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
V100 (Colab High RAM) (vs+CUDA) | 9.8 | 4.7 | 2.1

IFRNet (small model) | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
V100 (Colab High RAM / 8CPU) (vs+x264+FrameEval) | 78 | 47 | 23

IFRNet (large model) | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
V100 (Colab High RAM / 8CPU) (vs+x264+FrameEval) | ? | ? | 15

DPIR | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | 54 | 24.4

SCUNet | 480p | 720p | 1080p
-------- | ---- | ---- | ----
4090 (12 vs threads) | 10 | ? | ?

ST-MFNet | 480p | 720p | 1080p
-------- | ---- | ---- | ----
1070ti | 1.6 | OOM | OOM

<div id='license'/>

## License

This code uses code from other repositories, but the code I wrote myself is under BSD3.
