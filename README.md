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
   * [ncnn](#ncnn)
       * [If you have errors installing ncnn whl files with pip](#pip-error)
       * [Rife ncnn C++](#rife-ncnn-c)
       * [RealSR / ESRGAN ncnn](#sr-ncnn)
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
- [Rife4 with HolyWu/vs-rife](https://github.com/HolyWu/vs-rife/)
- RIFE ncnn with [styler00dollar/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan) and [HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan)
- [SwinIR with HolyWu/vs-swinir](https://github.com/HolyWu/vs-swinir)
- [Sepconv (enhanced) with sniklaus/revisiting-sepconv](https://github.com/sniklaus/revisiting-sepconv/)
- EGVSR with [Thmen/EGVSR](https://github.com/Thmen/EGVSR) and [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- BasicVSR++ with [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- RealBasicVSR with [ckkelvinchan/RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR)
- RealCUGAN with [bilibili/ailab](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)
- FILM with [google-research/frame-interpolation](https://github.com/google-research/frame-interpolation)
- PAN with [zhaohengyuan1/PAN](https://github.com/zhaohengyuan1/PAN)
- IFRNet with [ltkong218/IFRNet](https://github.com/ltkong218/IFRNet)
- M2M with [feinanshan/M2M_VFI](https://github.com/feinanshan/M2M_VFI)
- IFUNet with [98mxr/IFUNet](https://github.com/98mxr/IFUNet/)
- eisai with [ShuhongChen/eisai-anime-interpolator](https://github.com/ShuhongChen/eisai-anime-interpolator)
- SCUNet with [cszn/SCUNet](https://github.com/cszn/SCUNet)
- GMFupSS with [98mxr/GMFupSS](https://github.com/98mxr/GMFupSS)
- ST-MFNet with [danielism97/ST-MFNet](https://github.com/danielism97/ST-MFNet)
- VapSR with [zhoumumu/VapSR](https://github.com/zhoumumu/VapSR)
- GMFSS_union with [98mxr/GMFSS_union](https://github.com/98mxr/GMFSS_union)
- AI scene detection with [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [snap-research/EfficientFormer (EfficientFormerV2)](https://github.com/snap-research/EfficientFormer), [lucidrains/TimeSformer-pytorch](https://github.com/lucidrains/TimeSformer-pytorch) and [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)

Also used:
- TensorRT C++ inference with [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- ddfi with [Mr-Z-2697/ddfi-rife](https://github.com/Mr-Z-2697/ddfi-rife) (auto dedup-duplication, not an arch)
- nix with [lucasew/nix-on-colab](https://github.com/lucasew/nix-on-colab)
- custom ffmpeg with [markus-perl/ffmpeg-build-script](https://github.com/markus-perl/ffmpeg-build-script)
- lsmash with [AkarinVS/L-SMASH-Works](https://github.com/AkarinVS/L-SMASH-Works)
- wwxd with [dubhater/vapoursynth-wwxd](https://github.com/dubhater/vapoursynth-wwxd)
- scxvid with [dubhater/vapoursynth-scxvid](https://github.com/dubhater/vapoursynth-scxvid)

Model | ESRGAN | SRVGGNetCompact | Rife | SwinIR | Sepconv | EGVSR | BasicVSR++ | Waifu2x | RealBasicVSR | RealCUGAN | FILM | DPIR | PAN | IFRNet | M2M | IFUNet | eisai | SCUNet | GMFupSS | ST-MFNet | VapSR | GMFSS_union
---  | ------- | --------------- | ---- | ------ | ------- | ----- | ---------- | ------- | ------------ | --------- | ---- | ---- | --- | ------ | --- | ------ | ----- | ------ | ---- | ---- | --- | ---
CUDA | - | - | yes ([rife40](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view), [rife41](https://drive.google.com/file/d/1CPJOzo2CHr8AN3GQCGKOKMVXIdt1RBR1/view)) | [yes](https://github.com/HolyWu/vs-swinir/tree/master/vsswinir) | [yes](http://content.sniklaus.com/resepconv/network-paper.pytorch) | [yes](https://github.com/Thmen/EGVSR/raw/master/pretrained_models/EGVSR_iter420000.pth) | [yes](https://github.com/HolyWu/vs-basicvsrpp/releases/tag/model) | - | [yes](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) | [yes](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD) | [yes](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy) | - | [yes](https://github.com/zhaohengyuan1/PAN/tree/master/experiments/pretrained_models) | [yes](https://www.dropbox.com/sh/hrewbpedd2cgdp3/AADbEivu0-CKDQcHtKdMNJPJa?dl=0) | [yes](https://drive.google.com/file/d/1dO-ArTLJ4cMZuN6dttIFFMLtp4I2LnSG/view) | [yes](https://drive.google.com/file/d/1psrM4PkPhuM2iCwwVngT0NCtx6xyiqXa/view) | [yes](https://drive.google.com/drive/folders/1AiZVgGej7Tpn95ats6967neIEPdShxWy) | [yes](https://github.com/cszn/SCUNet/blob/main/main_download_pretrained_models.py) | [yes](https://github.com/98mxr/GMFupSS/tree/main/train_log) | [yes](https://drive.google.com/file/d/1s5JJdt5X69AO2E2uuaes17aPwlWIQagG/view) | - | yes ([vanilla](https://drive.google.com/file/d/1AsA7a4HNR4RjCeEmNUJWy5kY3dBC-mru/view) / [wgan](https://drive.google.com/file/d/1GAp9DljP1RCQXz0uu_GNn751NBMEQOUB/view))
TensorRT | yes (torch_tensorrt / C++ TRT) | yes (onnx_tensorrt / C++ TRT) [v2](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/RealESRGANv2_v1.7z), [v3](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/RealESRGANv3_v1.7z) | yes | - | - | - | - | [yes (C++ TRT)](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/waifu2x_v3.7z) | - | [yes (C++ TRT)](https://github.com/AmusementClub/vs-mlrt/releases/download/v9.2/models.v9.2.7z) | - | [yes (C++ TRT)](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/dpir_v3.7z) | - | - | - | - | - | - | - | - | [yes (C++ TRT)](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models) | -
ncnn | yes, but compile yourself ([realsr ncnn models](https://github.com/nihui/realsr-ncnn-vulkan/tree/master/models)) | yes, but compile yourself ([2x](https://files.catbox.moe/u62vpw.tar)) | [yes](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan/tree/master/models) | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | -

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
```bash
# install docker, command for arch
yay -S docker nvidia-docker nvidia-container-toolkit docker-compose

# Download prebuild image from dockerhub (recommended)
docker pull styler00dollar/vsgan_tensorrt:latest

# Build docker manually
# This step is not needed if you already downloaded the docker and is only needed if yo
# want to build it from scratch. Keep in mind that you need to set env variables in windows differently and
# this command will only work in linux. Run that inside that directory
DOCKER_BUILDKIT=1 docker build -t styler00dollar/vsgan_tensorrt:latest .
# If you want to rebuild from scratch or have errors, try to build without cache
DOCKER_BUILDKIT=1 docker build --no-cache -t styler00dollar/vsgan_tensorrt:latest . 

# run the docker with docker-compose
# go into the vsgan folder, inside that folder should be compose.yaml, run this command
# you can adjust folder mounts in the yaml file
# afterwards the vsgan folder will be mounted under `/workspace/tensorrt` and you can navigate 
# into it with `cd tensorrt`
docker-compose run --rm vsgan_tensorrt

# run docker with the sh startup script (linux)
sh start_docker.sh

# run docker manually
# the folderpath before ":" will be mounted in the path which follows afterwards
# contents of the vsgan folder should appear inside /workspace/tensorrt
docker run --privileged --gpus all -it --rm -v /home/vsgan_path/:/workspace/tensorrt styler00dollar/vsgan_tensorrt:latest

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

# Models are outside of docker image to minimize download size and will be downloaded on demand if you run code.
# If you want specific models you can look in https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models 
# or use the download scripts to get all of them. Models are expected to be placed under models/
```

If docker does not want to start, try this before you use docker:
```bash
# fixing docker errors
sudo systemctl start docker
sudo chmod 666 /var/run/docker.sock
```
Windows is mostly similar, but the path needs to be changed slightly:
```
Example for C://path
docker run --privileged --gpus all -it --rm -v /mnt/c/path:/workspace/tensorrt vsgan_tensorrt:latest
docker run --privileged --gpus all -it --rm -v //c/path:/workspace/tensorrt vsgan_tensorrt:latest
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

## Video guide (depricated)

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
You need to convert onnx models into engines. You need to do that on the same system where you want to do inference. Download onnx models from [here]( https://github.com/AmusementClub/vs-mlrt/releases/download/v7/models.v7.7z) or from [my Github page](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models). You can technically just use any ONNX model you want or convert a pth into onnx with [convert_esrgan_to_onnx.py](https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/convert_esrgan_to_onnx.py). Inside the docker, you do
```
trtexec --fp16 --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --buildOnly
```
Be aware that DPIR (color) needs 4 channels.
```
trtexec --fp16 --onnx=dpir_drunet_color.onnx --minShapes=input:1x4x8x8 --optShapes=input:1x4x720x1280 --maxShapes=input:1x4x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --buildOnly
```
Rife needs 8 channels.
```
trtexec --fp16 --onnx=rife.onnx --minShapes=input:1x8x64x64 --optShapes=input:1x8x720x1280 --maxShapes=input:1x8x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --buildOnly
```
and put that engine path into `inference_config.py`. Only do FP16 if your GPU does support it. 

**Warnings**: 
- You need to use the FP32 onnx, even if you want FP16, specify `--fp16` for FP16.
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

![IMG](https://github.com/Mr-Z-2697/ddfi/blob/main/example/ddfi.webp?raw=true)
![IMG](https://github.com/Mr-Z-2697/ddfi/blob/main/example/simp.webp?raw=true)

To use it, first you need to edit `ddfi.py` to select your interpolator of choise and then also apply the desired framerate. The official code uses 8x and I suggest you do so too. Small example:
```python
clip = core.misc.SCDetect(clip=clip, threshold=0.100)
clip = core.rife.RIFE(clip, model=9, sc=True, skip=False, multiplier=8)

clip = core.vfrtocfr.VFRToCFR(
    clip, os.path.join(tmp_dir, "tsv2nX8.txt"), 192000, 1001, True
) # 23.97 * 8
``` 

Afterwards, you need to use `deduped_vfi.py` similar to how you used `main.py`. Adjust paths and file extention.


<div id='ncnn'/>

## ncnn
If you have and AMD gpu, then you can at least use ncnn on your own system. The docker includes ncnn functionality.

**WARNING: It seems like some videos result in a broken output. For some reason a certain `webm` video produced very weird results, despite it working with other (non-ncnn) models. If you encounter this, just mux to a mkv with `ffmpeg -i input.webm -c copy output.mkv` and it should work properly again.**

Instructions for Manjaro:
```bash
yay -S vapoursynth-git ffms2 ncnn

# nvidia
yay -S nvidia-utils
# amd
yay -S vulkan-radeon
or
yay -S vulkan-amdgpu-pro
```

<div id='pip-error'/>

#### If you have errors installing ncnn whl files with pip:
It seems like certain pip versions are broken and will not allow certain ncnn whl files to install properly. If you have install erorrs, either run the install with `sudo` or manually upgrade your pip with
```
wget https://bootstrap.pypa.io/get-pip.py -O ./get-pip.py
python ./get-pip.py
python3 ./get-pip.py
``` 
`pip 21.0` is confirmed by myself to be broken.

<div id='rife-ncnn-c'/>

#### Rife ncnn C++ (recommended)
I forked [HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan) and added my own models in [styler00dollar/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan). For the full experience you need to get VMAF and misc.
```bash
# VMAF
wget https://github.com/Netflix/vmaf/archive/refs/tags/v2.3.1.tar.gz && \
  tar -xzf  v2.3.1.tar.gz && cd vmaf-2.3.1/libvmaf/ && \
  meson build --buildtype release && ninja -C build && \
  ninja -C build install

git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF && cd VapourSynth-VMAF && meson build && \
  ninja -C build && ninja -C build install

# MISC
git clone https://github.com/vapoursynth/vs-miscfilters-obsolete && cd vs-miscfilters-obsolete && meson build && \
  ninja -C build && ninja -C build install

# RIFE
git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-RIFE-ncnn-Vulkan && cd VapourSynth-RIFE-ncnn-Vulkan && \
  git submodule update --init --recursive --depth 1 && meson build && ninja -C build && ninja -C build install
```

<div id='rife-ncnn-python'/>

<div id='sr-ncnn'/>

#### RealSR / ESRGAN ncnn:
You can install precompiled whl files from [here](https://github.com/styler00dollar/realsr-ncnn-vulkan-python/releases/tag/v1a). If you want to compile it, visit [styler00dollar/realsr-ncnn-vulkan-python](https://github.com/styler00dollar/realsr-ncnn-vulkan-python).
```bash
sudo pacman -S base-devel vulkan-headers vulkan-icd-loader vulkan-devel
pip install [URL for whl]
```

Any ESRGAN model will work with this (aside RealESRGAN 2x because of pixelshuffle), when you have the fitting param file. Make sure the input is called "data" and output is "output".

If you want to convert a normal pth to ncnn, you need to do `pth->onnx->ncnn(bin/param)`. For the first step you can use `torch.onnx` and for the second one you can use [this website](https://convertmodel.com/).

<div id='waifu-ncnn'/>

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
- 4090 data fluctuating due to teamviewer cpu load and uses 11900k.
- 4090² uses 5950x.
- 4090³ uses 13900k.

ⓘ means that model not public yet

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
4090 (vs+TesnorRT8.4GA+opset16+12 vs threads)| 135 | 59 | 25
4090 (vs+TesnorRT8.4GA+opset16+12 vs threads+ffv1) | 155 | 72 | 35
4090 (vs+TensorRT8.4GA+opset16+12 vs threads+thread_queue_size) | 200 | 91 | X
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
4090²(2) (TensorRT8.4GA+vs_threads=4+num_streams=4+opset16+fp16) | ? | ? | ? / 55.1*
4090²(2) (TensorRT8.4GA+vs_threads=4+num_streams=4+opset16+int8) | ? | ? | ? / 57.7*
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
4090 (vs+TensorRT8.4GA+ffmpeg+C++ TRT+vs_threads=12+num_streams=6+no tiling+opset13) | 117 | 53 | 24
4090 (vs+TensorRT8.4GA+ffmpeg+C++ TRT+vs_threads=12+num_streams=5+no tiling+opset13+int8) | ? | ? | 17
4090 (vs+TensorRT8.4GA+ffmpeg+C++ TRT+vs_threads=12+num_streams=5+no tiling+opset13+int8+ffv1) | 132 | 61 | 29
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

ESRGAN 2x (64mb) (23b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
4090 (C++ TensorRT8.4GA+ffmpeg+int8+12 vs threads+4 num_streams+fp16) | ? / 6.1* | ? / ? | ? / ?
4090 (C++ TensorRT8.4GA+ffmpeg+int8+12 vs threads+1 num_streams+int8) | ? / 17.4* | ? / 7.1* | ? / 3.1*

Note: The offical RealESRGAN repository uses 6b (6 blocks) for the anime model.

RealESRGAN (4x) (6b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=2) | ? | 1.7 | 0.75
V100 (Colab High RAM) (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=1+no tiling) | 6.82 | 3.15 | OOM (OpenCL) 
V100 (Colab High RAM) (vs+TensorRT8.2GA+x264+C++ TRT+num_streams=1+no tiling) | ? | ? | 1.39
A100 (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=3+no tiling) | 14.65 | 6.74 | 2.76
3090² (C++ TRT+vs_threads=20+num_threads=2+no tiling+opset14) | 11 | 4.8 | 2.3
2x3090² (C++ TRT+vs_threads=10+num_threads=2+no tiling+opset14) | 22 | 9.5 | 4.2
4090 (C++ TensorRT8.4GA+ffmpeg+12 vs threads+1 num_streams+ffv1+opset16+fp16) | 19 / 19* (2 streams) | ? | ?
4090 (C++ TensorRT8.4GA+ffmpeg+12 vs threads+1 num_streams+ffv1+opset16+int8) | 34 (4 streams) / 50* (6 streams) | ? / ? | ? / 5.7* (1 stream)
4090³ (C++ TensorRT8.5+vs_threads=4+num_streams=1+fp16+(--heuristic) | ? | ? / 6.9* | ? / 3.1*
4090³ (C++ TensorRT8.5+vs_threads=4+num_streams=1+fp16) | ? | ? / 6.9* | ? / 3.1*

RealESRGAN (2x) (6b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
1070ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=1+no tiling+opset15) | 0.9 | 0.8 | 0.3
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=1) | ? | 3.12 | 1.4
V100 (Colab High RAM / 8CPU) (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=3+no tiling+opset15) | 5.09 | 4.56 | 2.02
V100 (Colab High RAM / 8CPU) (vs+TensorRT8.2GA+ffmpeg+C++ TRT+num_streams=3+no tiling+opset15) | 5.4 | 4.8 | 2.2
3090² (C++ TRT+vs_threads=20+num_threads=6+no tiling+opset16) (+dropout) | 13 | 5.8 | 2.7
2x3090² (C++ TRT+vs_threads=20+num_threads=6+no tiling+opset16) (+dropout) | 26 | 11 | 5.3
4090 (C++ TRT+TensorRT8.4GA+vs_threads=6+num_threads=6+no tiling+opset16+"--best") (+dropout) | ? | ? | ? / 12*

RealESRGAN (2x) (3b+64nf+dropout)ⓘ | 480p | 720p | 1080p
------------  | ---  | ---- | ------
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=2) | ? | 5.69 | 2.64
V100 (Colab High RAM / 8CPU) (vs+TensorRT8.4GA+ffmpeg+C++ TRT+num_streams=4+no tiling+opset15) | 10 | 9.4 | 4.2
3090² (C++ TRT+vs_threads=20+num_threads=6+no tiling+opset15) | 24 | 11 | 5.2
2x3090 (C++ TRT+vs_threads=20+num_threads=6+no tiling+opset15) | 51 | 23 | 10

Rife4.6 technically is fastmode=True, since contextnet/unet was removed.

Rife4+vs (fastmode False, ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (vs+ffmpeg+ModifyFrame) | 61 | 30 | 15
3060ti (vs+ffmpeg+ModifyFrame) | ? | 45 | 24

Rife4+vs (fastmode False, ensemble True) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti Python (vs+ffmpeg+ModifyFrame) | 27 | 13 | 9.6
1070ti C++ NCNN | ? | ? | 10
3060ti (vs+ffmpeg+ModifyFrame) | ? | 36 | 20 |
3090² (CUDA+vs_threads=20) | 70 | 52 | 27
3090² (C++ NCNN+vs_threads=20+ncnn_threads=8) | 137 | 65 | 31
V100 (Colab) (vs+ffmpeg+ModifyFrame) | 30 | 16 | 7.3
V100 (Colab High RAM) (vs+x264+ModifyFrame) | 48.5 | 33 | 19.2
V100 (Colab High RAM) (vs+x264+FrameEval) | 48.2 | 35.5 | 20.6
V100 (Colab High RAM) (vs+x265+FrameEval) | 15.2 | 9.7 | 4.6
V100 (Colab High RAM / 8CPU) (vs+x264+C++ NCNN (7 threads)) | 70 | 35 | 17
A100 (Colab) (vs+CUDA+ffmpeg+ModifyFrame) | 54 | 39 | 23
A100 (Colab) (jpg+CUDA+ffmpeg+ModifyFrame) | ? | ? | 19.92 (14 Threads)
4090 (vs+CUDA+ffmpeg+FrameEval+12 vs threads) (rife40) | 61 | 61 | 36
4090 (ncnn+8 threads+12 vs threads) (rife4.0) | 254 | 130 | 60

Rife4+vs (fastmode True, ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti Python (ffmpeg+ModifyFrame) | 62 | 31 | 14
1070ti (C++ NCNN) (rife46) | ? | ? | 30
1070ti (TensorRT8.5+num_streams=3) (rife46) | ? | ? | 27
3060ti (CUDA+ffmpeg+ModifyFrame) | ? | 66 | 33 |
3090² (CUDA+ffmpeg+FrameEval+vs_threads=20) | 121 | 80 | 38
3090² (C++ NCNN+vs_threads=20+ncnn_threads=8) | 341 | 142 | 63
3090³ (TensorRT8.5+6 vs_threads) | ? / 331.9* (9 streams) | ? / 275.3* (7 streams) | ? / 166.3* (7 streams)
4090 (ncnn+8 threads+12 vs threads) (rife4.0) | 470 | 198 | 98
4090 (ncnn+8 threads+12 vs threads) (rife4.4) | - | - | 98
4090 (ncnn+8 threads+12 vs threads+ffv1) (rife4.4) |- |	- |	129 / 128*
4090 (ncnn+8 threads+12 vs threads) (rife4.6) | 455 | 215 | 100 / 136*
4090² (ncnn+2 threads+4 vs threads+ffmpeg (ultrafast)) (rife4.6) | ? | ? | 164
4090 (TensorRT8.5+num_streams 8+num_threads=6+stacking method) (rife46) | ? | ? | ? / 146*
4090 (TensorRT8.5+num_streams 8+num_threads=6+int8+ffv1+stacking method) (rife46)| ? | ? | 123 / 156*
4090³ (TensorRT8.5+vs_threads=4+fp16) (rife46) | ? | ? / 541* (num_streams=14) | ? / 288* (num_streams=10)
V100 (Colab) (ffmpeg+ModifyFrame) | 34 | 17 | 7.6
V100 (Colab High RAM / 8CPU) (vs+x264+FrameEval) | 64 | 43 | 25
V100 (Colab High RAM / 8CPU) (vs+x264+C++ NCNN (8 threads)) | 136 | 65 | 29
A100 (Colab) (ffmpeg+ModifyFrame) | 92 | 56 | 29
A100 (Colab/12CPU) (ncnn+8 threads+12 vs threads) (rife40) | 208 | 103 | 46
A100 (Colab/12CPU) (ncnn+8 threads+12 vs threads+ffv1) (rife40) | 87 | 97 | 48
6700xt (vs_trheads=4, num_threads=2) | ? / 258.5* | ? / 122.4* | ? / 55.8*

Rife4+vs (fastmode True, ensemble True) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (PyTorch+ffmpeg+ModifyFrame) | 41 | 20 | 9.8 
1070ti (C++ NCNN) (rife46) | ? | ? | 16
1070ti (TensorRT8.5+num_streams=2) (rife46) | ? | ? | 14
3060ti (ffmpeg+ModifyFrame) | ? | 49 | 24 
3090¹ (ffmpeg+ModifyFrame) | ? | 90.3 | 45
4090 (vs+CUDA+ffmpeg+FrameEval) (rife46) | 84 | 80 | 41
4090 (ncnn+8 threads+12 vs threads) (rife4.6) | 280 | 165 | 76
4090 (ncnn+8 threads+12 vs threads) (rife4.6+ffv1) | 222 | 162 | 80
4090³ (TensorRT8.5+vs_threads=4+fp16) (rife46) | ? | 320 / 401.6* (num_streams=14) | 160 / 207* (num_streams=10)
A100 (Colab/12CPU) (ncnn+8 threads+12 vs threads) (rife46) | 154 | 86 | 43
A100 (Colab/12CPU) (ncnn+8 threads+12 vs threads+ffv1) (rife46) | 86 | 86 | 43
6700xt (vs_trheads=4, num_threads=2) | ? / 129.7* | ? / 60.4* | ? / 28*

GMFupSS | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
T4 (Colab / 8CPU) | 8.1 | 3.4 | 1.3
T4 (Colab / 8CPU) (partial fp16) | 13 | 5.2 | 2.3
A100 (Colab / 12CPU) | 23 | 14 | 6.2
4090 (12 vs threads) | 27 | 22 | 8.6
4090 (12 vs threads + thread_queue_size) | 32 | 21 | 8.6
4090³ (num_threads=4, partial fp16) | ? | ? / 34.4* | ? / 13.1*

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
3090¹ (TensorRT8+C++ TRT+ffmpeg+vs threads=7+num_streams=5) | ? | ? | 16
4090 (num_streams=13+12 vs threads) | 121 | 52 | 23
4090 (num_streams=13+12 vs threads+thread_queue_size) | 121 | 54 | 23
4090 (num_streams=13+12 vs threads+ffv1+thread_queue_size) | 121 | 55 | 25
4090 (num_streams=13+12 vs threads+ffv1+int8) | ? | ? | 52
4090 (num_streams=13+12 vs threads+ffv1+int8+thread_queue_size) | ? | ? | 44

SCUNet | 480p | 720p | 1080p
-------- | ---- | ---- | ----
4090 (12 vs threads) | 10 | ? | ?

ST-MFNet | 480p | 720p | 1080p
-------- | ---- | ---- | ----
1070ti | 1.6 | OOM | OOM

<div id='license'/>

## License

This code uses code from other repositories, but the code I wrote myself is under BSD3.
