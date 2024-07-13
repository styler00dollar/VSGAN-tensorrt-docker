# VSGAN-tensorrt-docker

Repository to use super resolution models and video frame interpolation models and also trying to speed them up with TensorRT. This repository contains the fastest inference code that you can find, at least I am trying to archive that. Not all codes can use TensorRT due to various reasons, but I try to add that if it works. Further model architectures are planned to be added later on.

Table of contents
=================

<!--ts-->
   * [Usage](#usage)
   * [Usage example](#usage-example)
   * [Deduplicated inference](#deduplicated)
   * [Shot Boundary Detection](#shot-boundry)
   * [vs-mlrt (C++ TRT)](#vs-mlrt)
       * [multi-gpu](#multi-gpu)
   * [ddfi](#ddfi)
   * [VFR (variable refresh rate)](#vfr)
   * [Color transfer](#color)
   * [Benchmarks](#benchmarks)
   * [License](#license)
<!--te-->

-------

Currently working networks:
- [Rife4 with HolyWu/vs-rife](https://github.com/HolyWu/vs-rife/) and [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) ([rife4.0](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view) [rife4.1](https://drive.google.com/file/d/1CPJOzo2CHr8AN3GQCGKOKMVXIdt1RBR1/view) [rife4.2](https://drive.google.com/file/d/1JpDAJPrtRJcrOZMMlvEJJ8MUanAkA-99/view)
[rife4.3](https://drive.google.com/file/d/1xrNofTGMHdt9sQv7-EOG0EChl8hZW_cU/view) [rife4.4](https://drive.google.com/file/d/1eI24Kou0FUdlHLkwXfk-_xiZqKaZZFZX/view) [rife4.5](https://drive.google.com/file/d/17Bl_IhTBexogI9BV817kTjf7eTuJEDc0/view) [rife4.6](https://drive.google.com/file/d/1EAbsfY7mjnXNa6RAsATj2ImAEqmHTjbE/view) [rife4.7.1](https://drive.google.com/file/d/1s2zMMIJrUAFLexktm1rWNhlIyOYJ3_ju/view) [rife4.8.1](https://drive.google.com/file/d/1wZa3SyegLPUwBQWmoDLM0MumWd2-ii63/view)
[rife4.9.2](https://drive.google.com/file/d/1UssCvbL8N-ty0xIKM5G5ZTEgp9o4w3hp/view) [rife4.10.1](https://drive.google.com/file/d/1WNot1qYBt05LUyY1O9Uwwv5_K8U6t8_x/view) [rife4.11.1](https://drive.google.com/file/d/1Dwbp4qAeDVONPz2a10aC2a7-awD6TZvL/view) [rife4.12.2](https://drive.google.com/file/d/1ZHrOBL217ItwdpUBcBtRE3XBD-yy-g2S/view) [rife4.12 lite](https://drive.google.com/file/d/1KoEJ5x6aisxOpkhdNptsGUrL3yf4iy8b/view) [rife4.13.2](https://drive.google.com/file/d/1mj9lH6Be7ztYtHAr1xUUGT3hRtWJBy_5/view) [rife4.13 lite](https://drive.google.com/file/d/1l3lH9QxQQeZVWtBpdB22jgJ-0kmGvXra/view) [rife4.14](https://drive.google.com/file/d/1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy/view) [rife4.14 lite](https://drive.google.com/file/d/1eULia_onOtRXHMAW9VeDL8N2_7z8J1ba/view) [rife4.15](https://drive.google.com/file/d/1xlem7cfKoMaiLzjoeum8KIQTYO-9iqG5/view) [rife4.17](https://drive.google.com/file/d/1962p_lEWo_kLTEynarNaRYRNVdaiQG2k/view) [rife4.18](https://drive.google.com/file/d/1octn-UVuEjXa_HlsIUbNeLTTvYCKbC_s/view) [rife4.19-beta](https://drive.google.com/file/d/1Wf09iMyTrvVdG7oG3blVAEHDdLU2dhW2/view))

- RealCUGAN with [bilibili/ailab](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)
- GMFupSS with [98mxr/GMFupSS](https://github.com/98mxr/GMFupSS)
- GMFSS_union with [HolyWu version](https://github.com/HolyWu/vs-gmfss_union)
- GMFSS_Fortuna and GMFSS_Fortuna_union with [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna)
- SRVGGNetCompact with [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- Model based shot boundary detection with [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [snap-research/EfficientFormer (EfficientFormerV2)](https://github.com/snap-research/EfficientFormer) and [wentaozhu/AutoShot](https://github.com/wentaozhu/AutoShot)

Also used:
- TensorRT C++ inference and python script usage with [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- ddfi with [Mr-Z-2697/ddfi-rife](https://github.com/Mr-Z-2697/ddfi-rife) (auto dedup-duplication, not an arch)
- custom ffmpeg with [styler00dollar/ffmpeg-static-arch-docker](https://github.com/styler00dollar/ffmpeg-static-arch-docker)
- lsmash with [AkarinVS/L-SMASH-Works](https://github.com/AkarinVS/L-SMASH-Works)
- bestsource with [vapoursynth/bestsource](https://github.com/vapoursynth/bestsource)
- trt precision check and upscale frame skip with [mafiosnik777/enhancr](https://github.com/mafiosnik777/enhancr)

Model | Rife | GMFupSS | GMFSS_union | GMFSS_Fortuna / GMFSS_Fortuna_union
----- | ---- | ------- | ----------- | -----------------------------------
CUDA | yes (4.0-4.12) | [yes](https://github.com/98mxr/GMFupSS/tree/main/train_log) | yes ([vanilla](https://drive.google.com/file/d/1AsA7a4HNR4RjCeEmNUJWy5kY3dBC-mru/view) / [wgan](https://drive.google.com/file/d/1GAp9DljP1RCQXz0uu_GNn751NBMEQOUB/view)) | yes ([base](https://drive.google.com/file/d/1BKz8UDAPEt713IVUSZSpzpfz_Fi2Tfd_/view) / [union](https://drive.google.com/file/d/1Mvd1GxkWf-DpfE9OPOtqRM9KNk20kLP3/view))
TensorRT | yes (4.0-4.19, skipped some lite models) | - | - | -

Further stuff that can use TensorRT via [mlrt](https://github.com/AmusementClub/vs-mlrt) with onnx is for example [Real-ESRGAN / SRVGGNetCompact](https://github.com/xinntao/Real-ESRGAN), [SAFMN](https://github.com/sunny2109/SAFMN), [DPIR](https://github.com/cszn/DPIR), [Waifu2x](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/waifu2x_v3.7z), [real-cugan](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD), [apisr](https://github.com/kiteretsu77/apisr), [AnimeJaNai](https://github.com/the-database/mpv-upscale-2x_animejanai), [ModernSpanimation](https://github.com/TNTwise/Models/releases/tag/2x_ModernSpanimationV1) and [AniScale](https://github.com/Sirosky/Upscale-Hub). Onnx files can be found [here](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models).

Some important things:
- If you are on Windows, install all the latest updates first, otherwise wsl won't work properly. 21H2 minimum.
- Do not use `webm` video, webm is often broken. It can work, but don't complain about broken output afterwards. I would suggest to render webm into mp4 or mkv.
- Only use ffmpeg to determine if video is variable framerate (vfr) or not. Other programs do not seem reliable.
- Processing vfr video is dangerous, but you can try to use fpsnum and fpsden. Either use these params or render the input video into constant framerate (crf).
- Colabs have a weak cpu, you should try `x264` with `--opencl`. (A100 does not support NVENC and such)

<div id='usage'/>

## Usage
Get CUDA12.1 and latest Nvidia drivers. After that, follow the following steps:

**WARNING FOR WINDOWS USERS: Docker Desktop `4.17.1` is broken. I confirmed that [4.25.0](https://desktop.docker.com/win/main/amd64/126437/Docker%20Desktop%20Installer.exe) should work. Older tested versions are [4.16.3](https://desktop.docker.com/win/main/amd64/96739/Docker%20Desktop%20Installer.exe) or [4.17.0](https://desktop.docker.com/win/main/amd64/99724/Docker%20Desktop%20Installer.exe). I would recommend to use `4.25.0`. `4.17.1` results in Docker not starting which is mentioned in [this issue](https://github.com/styler00dollar/VSGAN-tensorrt-docker/issues/34).**

**ANOTHER WARNING FOR PEOPLE WITHOUT `AVX512`: Instead of using `styler00dollar/vsgan_tensorrt:latest`, which I build with my 7950x and thus with all AVX, use `styler00dollar/vsgan_tensorrt:latest_no_avx512` in `compose.yaml` to avoid `Illegal instruction (core dumped)` which is mentioned in [this issue](https://github.com/styler00dollar/VSGAN-tensorrt-docker/issues/48).**

**AND AS A FINAL INFO, `Error opening input file pipe:` IS NOT A REAL ERROR MESSAGE. That means invalid data got piped into ffmpeg and can be piped error messages for example. To see the actual error messages and what got piped, you can use `vspipe -c y4m inference.py -`.**

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
- `latest`: Default docker with everything. Trying to keep everything updated and fixed.
- `latest_no_avx512` is for cpus without avx512 support, otherwise it just crashes if you try to run avx512 binaries on cpus without such support. Use this if your cpu does not support all instruction sets.
- `minimal`: Bare minimum to run `ffmpeg`, `mlrt` and a few video readers.

| docker image  | compressed download | extracted container | short description |
| ------------- | ------------------- | ------------------- | ----------------- |
| styler00dollar/vsgan_tensorrt:latest | 8gb | 15gb | default latest trt9.3
| styler00dollar/vsgan_tensorrt:latest_no_avx512 | 8gb | 15gb | default latest trt9.3 without avx512
| styler00dollar/vsgan_tensorrt:minimal | 4gb | 8gb | trt8.6 + ffmpeg + mlrt + ffms2 + lsmash + bestsource
| styler00dollar/vsgan_tensorrt:trt10.0 | 8gb | 15gb | trt10.0 (not recommended, rife broken)

Piping usage:
```
# you can use it in various ways, ffmpeg example
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv -y

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
and then afterwards edit `inference_config.py`.

Small example for upscaling with TensorRT:

```python
import sys
import os

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core.num_threads = 8

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


def inference_clip(video_path="", clip=None):
    clip = core.bs.VideoSource(source=video_path)

    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")  # RGBS means fp32, RGBH means fp16
    clip = core.trt.Model(
        clip,
        engine_path="/workspace/tensorrt/2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp.engine",  # read readme on how to build engine
        num_streams=2,
    )
    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")  # you can also use YUV420P10 for example

    return clip
```

Small example for rife interpolation with TensorRT without scene change detection:

```python
import sys
import vapoursynth as vs
from src.rife import RIFE
from src.vfi_inference import vfi_inference

sys.path.append("/workspace/tensorrt/")
core = vs.core
core.num_threads = 4

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


def inference_clip(video_path):
    clip = core.bs.VideoSource(source=video_path)

    clip = core.resize.Bicubic(
        clip, format=vs.RGBS, matrix_in_s="709"
    )  # RGBS means fp32, RGBH means fp16

    # interpolation
    clip = rife_trt(
        clip,
        multi=2,
        scale=1.0,
        device_id=0,
        num_streams=2,
        engine_path="/workspace/tensorrt/rife414_ensembleTrue_op18_fp16_clamp_sim.engine",  # read readme on how to build engine
    )

    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
```

Small example for PyTorch interpolation with rife without scene change detection:
```python
import sys
import vapoursynth as vs
from src.rife import RIFE
from src.vfi_inference import vfi_inference

sys.path.append("/workspace/tensorrt/")
core = vs.core
core.num_threads = 4

def inference_clip(video_path):
    clip = core.bs.VideoSource(source=video_path)

    clip = core.resize.Bicubic(
        clip, format=vs.RGBS, matrix_in_s="709"
    )  # RGBS means fp32, RGBH means fp16

    # interpolation
    model_inference = RIFE(
        scale=1, fastmode=False, ensemble=True, model_version="rife46", fp16=True
    )
    clip = vfi_inference(model_inference=model_inference, clip=clip, multi=2)

    clip = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
```

More examples in `custom_scripts/`.

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

<div id='deduplicated'/>

## Deduplicated inference
Calculate similarity between frames with [HomeOfVapourSynthEvolution/VapourSynth-VMAF](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF) and skip similar frames in interpolation tasks. The properties in the clip will then be used to skip similar frames.

```python
from src.rife_trt import rife_trt

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


# calculate metrics
def metrics_func(clip):
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    return core.vmaf.Metric(clip, offs1, 2)

def inference_clip(video_path):
    interp_scale = 2
    clip = core.bs.VideoSource(source=video_path)

    # ssim
    clip_metric = vs.core.resize.Bicubic(
        clip, width=224, height=224, format=vs.YUV420P8, matrix_s="709"  # resize before ssim for speedup
    )
    clip_metric = metrics_func(clip_metric)
    clip_orig = core.std.Interleave([clip] * interp_scale)

    # interpolation
    clip = rife_trt(
        clip,
        multi=interp_scale,
        scale=1.0,
        device_id=0,
        num_streams=2,
        engine_path="/workspace/tensorrt/rife414_ensembleTrue_op18_fp16_clamp_sim.engine",
    )

    # skip frames based on calculated metrics
    # in this case if ssim > 0.999, then copy frame
    clip = core.akarin.Select([clip, clip_orig], clip_metric, "x.float_ssim 0.999 >")

    return clip
```

There are multiple different metrics that can be used, but be aware that you may need to adjust the threshold metric value in `vfi_inference.py`, since they work differently. SSIM has a maximum of 1 and PSNR has a maximum of infinity. I would recommend leaving the defaults unless you know what you do.
```python
# 0 = PSNR, 1 = PSNR-HVS, 2 = SSIM, 3 = MS-SSIM, 4 = CIEDE2000
return core.vmaf.Metric(clip, offs1, 2)
```

<div id='shot-boundry'/>

## Shot Boundary Detection

Detection is implemented in various different ways. To use traditional scene change you can do:

```python
clip_sc = core.misc.SCDetect(
  clip=clip,
  threshold=0.100
)
```
Afterwards you can call `clip = core.akarin.Select([clip, clip_orig], clip_sc, "x._SceneChangeNext 1 0 ?")` to apply it.

Or use models like this. Adjust thresh to a value between 0 and 1, higher means to ignore with less confidence.

```python
clip_sc = scene_detect(
    clip,
    fp16=True,
    thresh=0.5,
    model=3,
)
```

**Warning: Keep in mind that different models may require a different thresh to be good.**

The rife models mean, that flow gets used as an additional input into the classification model. That should increase stability without major speed decrease. Models that are not linked will be converted later.

Available onnx files:
- efficientnetv2_b0 (256px) ([fp16](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientnetv2b0_17957_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx) [fp32](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientnetv2b0_17957_256_CHW_6ch_clamp_softmax_op17_sim.onnx))
- efficientnetv2_b0+rife46 (256px) ([fp16](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientnetv2b0+rife46_flow_1362_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx) [fp32](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientnetv2b0+rife46_flow_1362_256_CHW_6ch_clamp_softmax_op17_sim.onnx))
- efficientformerv2_s0 (224px) ([fp16](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx) [fp32](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_sim.onnx))
- efficientformerv2_s0+rife46 (224px) ([fp16](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientformerv2_s0+rife46_flow_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx) [fp32](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_efficientformerv2_s0+rife46_flow_84119_224_CHW_6ch_clamp_softmax_op17.onnx))
- swinv2_small (256px) ([fp16](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_swinv2_small_window16_10412_256_CHW_6ch_clamp_softmax_op17_fp16.onnx) [fp32](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_swinv2_small_window16_10412_256_CHW_6ch_clamp_softmax_op17.onnx))
- swinv2_small+rife46 (256px) ([fp16](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_swinv2_small_window16+rife46_flow_1814_256_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx) [fp32](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sc_swinv2_small_window16+rife46_flow_1814_256_84119_224_CHW_6ch_clamp_softmax_op17.onnx))

Other models I trained but are not available due to various reasons:
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
- TimeSformer
- maxvit_small
- maxvit_small+rife46
- regnetz_005
- repvgg_b0
- resnetrs50
- resnetv2_50
- rexnet_100

Interesting observations:
- Applying means/stds seemingly worsened results, despite people doing that as standard practise.
- Applying image augmentation worsened results.
- Training with higher batchsize made detections a little more stable, but maybe that was placebo and a result of more finetuning.

Comparison to traditional methods:
- [wwxd](https://github.com/dubhater/vapoursynth-wwxd) and [scxvid](https://github.com/dubhater/vapoursynth-scxvid) suffer from overdetection (at least in drawn animation).
- The json that [master-of-zen/Av1an](https://github.com/master-of-zen/Av1an) produces with `--sc-only --sc-method standard --scenes test.json` returns too little scene changes. Changing the method does not really influence a lot. Not reliable enough for vfi.
- I can't be bothered to [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) get working with vapousynth with FrameEval and by default it only works with video or image sequence as input. I may try in the future, but I don't understand why I cant just input two images.
- `misc.SCDetect` seemed like the best traditional vapoursynth method that does currently exist, but I thought I could try to improve. It struggles harder with similar colors and tends to skip more changes compared to methods.

Decided to only do scene change inference with ORT with TensorRT backend to keep code small and optimized.

Example usage:
```python
from src.scene_detect import scene_detect
from src.rife_trt import rife_trt

core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


clip_sc = scene_detect(
    clip,
    fp16=True,
    thresh=0.5,
    model=3,
)

clip = rife_trt(
    clip,
    multi=2,
    scale=1.0,
    device_id=0,
    num_streams=2,
    engine_path="/workspace/tensorrt/rife414_ensembleTrue_op18_fp16_clamp_sim.engine",
)

clip_orig = core.std.Interleave([clip_orig] * 2)  # 2 means interpolation factor here
clip = core.akarin.Select([clip, clip_orig], clip_sc, "x._SceneChangeNext 1 0 ?")
```

<div id='vs-mlrt'/>

## vs-mlrt (C++ TRT)
You need to convert onnx models into engines. You need to do that on the same system where you want to do inference. Download onnx models from [here]( https://github.com/AmusementClub/vs-mlrt/releases/download/v7/models.v7.7z) or from [my Github page](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models). You can technically just use any ONNX model you want or convert a pth into onnx with for example [convert_compact_to_onnx.py](https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/convert_compact_to_onnx.py). Inside the docker, you do one of the following commands:


Good default choice:
```
trtexec --bf16 --fp16 --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5
```
If you have the vram to fit the model multiple times, add `--infStreams`.
```
trtexec --bf16 --fp16 --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=4
```
DPIR (color) needs 4 channels.
```
trtexec --bf16 --fp16 --onnx=model.onnx --minShapes=input:1x4x8x8 --optShapes=input:1x4x720x1280 --maxShapes=input:1x4x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5
```
Rife v1 needs 8 channels. Setting `fasterDynamicShapes0805` since trtexec recommends it.
```
trtexec --bf16 --fp16 --onnx=model.onnx --minShapes=input:1x8x64x64 --optShapes=input:1x8x720x1280 --maxShapes=input:1x8x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --preview=+fasterDynamicShapes0805
```
Rife v2 needs 7 channels. Set the same shape everywhere to avoid build errors.
```
trtexec --bf16 --fp16 --onnx=model.onnx --minShapes=input:1x7x1080x1920 --optShapes=input:1x7x1080x1920 --maxShapes=input:1x7x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5
```

Put that engine path into `inference_config.py`.

**Warnings**:
- Only add `--bf16` if your GPU supports it, otherwise remove it. If model looks broken, remove `--fp16`.
- Cugan with 3x scale requires same MIN/OPT/MAX shapes.
- rvpV2 needs 6 channels, but does not support variable shapes.
- If you use the FP16 onnx you need to use `RGBH` colorspace, if you use FP32 onnx you need to use `RGBS` colorspace in `inference_config.py` .
- Engines are system specific, don't use across multiple systems.
- Don't use reuse engines for different GPUs.
- If you run out of memory, then you need to adjust the resolutions in that command. If your video is bigger than what you can input in the command, use tiling.
- If you get segfault, reduce `builderOptimizationLevel`. Change can change it down to 1 to speed up the engine building, but may result in worse speeds.
- If you set min, opt and max to the same resolution, it might result in a faster engine.

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

Example usage is in `custom_scripts/ddfi_rife_dedup_scene_change/`. As a quick summary, you need to do two processing passes. One pass to calculate metrics and another to use interpolation combined with VFRToCFR. You need to use `deduped_vfi.py` similar to how you used `main.py`.

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
ffmpeg -i video_input.mkv -fps_mode cfr -crf 10 -c:a copy video_out.mkv
```
or use my `vfr_to_cfr.py` to process a folder.

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
4090 (TRT9.3+num_streams=3+(fp16+bf16)+RGBH+op18) | ? | ? / 92.3* | ? / 41.5*
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

Note: The offical RealESRGAN-6b model uses 6 blocks for the anime model and uses the ESRGAN architecture.

RealESRGAN (4x) (6b+64nf) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=2) | ? | 1.7 | 0.75
V100 (Colab High RAM) (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=1+no tiling) | 6.82 | 3.15 | OOM (OpenCL)
V100 (Colab High RAM) (vs+TensorRT8.2GA+x264+C++ TRT+num_streams=1+no tiling) | ? | ? | 1.39
A100 (vs+TensorRT8.2GA+x264 (--opencl)+C++ TRT+num_streams=3+no tiling) | 14.65 | 6.74 | 2.76
3090² (C++ TRT+vs_threads=20+num_threads=2+no tiling+opset14) | 11 | 4.8 | 2.3
2x3090² (C++ TRT+vs_threads=10+num_threads=2+no tiling+opset14) | 22 | 9.5 | 4.2
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 8.8* | ? / 3.9*

Rife v2 refers to a custom implementation made by [WolframRhodium](https://github.com/WolframRhodium). I would recommend to avoid `int8` for 1080p, the warping looks a bit broken. `int8` seems usable with 720p and looks closer to `bf16`/`fp16`. TRT10 is slower than 9.3 and thus not recommended. Windows seems slower than Linux by quite a margin. Not all show major improvement with above 3 streams. There mostly seems to be no difference between level 3 and 5.

Rife4+vs (ensemble False) | 480p | 720p | 1080p
-------  | -------  | ------- | -------
Rife 4.6  | -------  | ------- | -------
4090 rife4.6 (Win11 vs-ncnn+num_streams=3+RGBS) | ? | ? | ? / 134.3*
4090 rife4.6 (Arch KDE vs-rife+TRT10 (level 5)+num_streams=3+RGBH) | ? | ? / 827.1* | ? / 357.9*
4090 rife4.6 (Win11 mlrt+TRT9.2 (level 3)+num_streams=3+RGBH) | ? | ? | ? / 294.5*
4090 rife4.6 (Win11 VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op18) | ? | ? | ? / 372.7*
4090 rife4.6 (Manjaro Gnome VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op18) | ? | ? / 1083.3* | ? / 469.9*
4090 rife4.6 v2 (Win11 mlrt+TRT9.2 (level 3)+num_streams=3+RGBH) | ? | ? | ? / 442.4*
4090 rife4.6 v2 (Win11 mlrt+TRT9.2 (level 3)+num_streams=8+RGBH) | ?  | ? | ? / 480.2*
4090 rife4.6 v2 (Arch KDE VSGAN+TRT9.3 (level 5)+num_streams=3+RGBH+op16 (fp16 converted mlrt onnx)) | ?  | ? / 1228.4* | ? / 511*
Steam Deck rife4.6 (ncnn+RGBS) | ? | ? / 19.2* | ? / 8.8*
Rife 4.15  | -------  | ------- | -------
4090 rife4.15 (Win11 vs-ncnn+num_streams=3+RGBS) | ? | ? | ? / 115.2*
4090 rife4.6 (Arch KDE vs-rife+TRT10 (level 5)+num_streams=3+RGBH) | ? | ? / 506.3* | ? / 204.2*
4090 rife4.15 (Win11 mlrt+TRT9.2 (level 3)+num_streams=3+RGBH) | ? | ? | ? / 237.7*
4090 rife4.15 (Win11 VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op19) | ? | ? | ? / 205*
4090 rife4.15 (Arch Gnome VSGAN (level 5)+TRT9.3+num_streams=3+(fp16+bf16)+RGBH+op19) | ? | ? | ? / 245.5*
4090 rife4.15 v2 (Win11 mlrt+TRT9.2 (level 3)+num_streams=3+RGBH) | ? | ? | ? / 276.8*
4090 rife4.15 v2 (Arch KDE VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op20) | ? | ? / 930.9* | ? / 360.1*
4090 rife4.15 v2 (Arch KDE VSGAN+TRT9.3 (level 5)+num_streams=3+(int8+fp16+bf16)+RGBH+op20) | ? | ? / 995.3* | ? / 424*
4090 rife4.15 v2 (Arch KDE VSGAN+TRT9.3 (level 5)+num_streams=8+(int8+fp16+bf16)+RGBH+op20) | ? | ? / 1117.6* | ? / 444.5*

Rife4+vs (ensemble True) | 480p | 720p | 1080p
-------  | -------  | ------- | -------
Rife 4.6  | -------  | ------- | -------
4090 rife4.6 (Win11 vs-ncnn+num_streams=3+RGBS) | ? | ? | ? / 89.5*
4090 rife4.6 (Arch KDE vs-rife+TRT10 (level 5)+num_streams=3+RGBH) | ? | ? / 649.6* | ? / 237.7*
4090 rife4.6 (Win11 mlrt+TRT9.3 (level 3)+num_streams=3) | ? | ? | ? / 226.7*
4090 rife4.6 (Win11 VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op18) | ? | ? | ? / 228.7*
4090 rife4.6 (Manjaro Gnome VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op18) | ? | ? / 671.4* | ? / 303.8*
4090 rife4.6 v2 (Win11 mlrt+TRT9.3 (level 3)+num_streams=3) | ? | ? | ? / 251.8*
4090 rife4.6 v2 (Arch KDE VSGAN (level 5)+TRT9.3+num_streams=3+RGBH+op16 (fp16 converted mlrt onnx)) | ?  | ? / 843.8* | ? / 346.2*
Rife 4.15  | -------  | ------- | -------
4090 rife4.15 (Win11 vs-ncnn+num_streams=3+RGBS) | ? | ? | ? / 67*
4090 rife4.15 (Arch KDE vs-rife+TRT10 (level 5)+num_streams=3+RGBH) | ? | ? / 339.6* | ? / 142.2*
4090 rife4.15 (Win11 mlrt+TRT9.2 (level 3)+num_streams=3+RGBH) | ? | ? | ? / 133.4*
4090 rife4.15 (Win11 VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op19) | ? | ? | ? / 139.8*
4090 rife4.15 (Manjaro Gnome VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op19) | ? | ? / 348.5* | ? / 149.6*
4090 rife4.15 v2 (Win11 mlrt+TRT9.2 (level 3)+num_streams=3+RGBH) | ? | ? | ? / 147.3*
4090 rife4.15 v2 (Arch KDE VSGAN+TRT9.3 (level 5)+num_streams=3+(fp16+bf16)+RGBH+op20) | ? | ? / 463.1* | ? / 181.3*
4090 rife4.15 v2 (Arch KDE VSGAN+TRT9.3 (level 5)+num_streams=3+(int8+fp16+bf16)+RGBH+op20) | ? | ? / 557.5* | ? / 210.6*

* Benchmarks made with [HolyWu version](https://github.com/HolyWu/vs-gmfss_union) with threading and partial TensorRT and without setting `tactic` to `JIT_CONVOLUTIONS` and `EDGE_MASK_CONVOLUTIONS` due to performance penalty. I added [a modified version](https://github.com/styler00dollar/vs-gmfss_union) as a plugin to VSGAN, but I need to add enhancements to my own repo later.

GMFSS_union | 480p | 720p | 1080p
-------- | ---- | ---- | ----
4090 (num_threads=8, num_streams=3, RGBH, TRT8.6, matmul_precision=medium) | ? | ? / 44.6* | ? / 15.5*

GMFSS_fortuna_union | 480p | 720p | 1080p
-------- | ---- | ---- | ----
4090 (num_threads=8, num_streams=2, RGBH, TRT8.6.1, matmul_precision=medium) | ? | ? / 50.4* | ? / 16.9*
4090 (num_threads=8, num_streams=2, RGBH, TRT8.6.1, matmul_precision=medium, @torch.compile(mode="default", fullgraph=True)) | ? | ? / 50.6* | ? / 17*

DPIR | 480p | 720p | 1080p
-------- | ---- | ---- | ----
4090 (TRT9.1+num_threads=4+num_streams=2+(fp16+bf16)+RGBH+op18) | ? | ? / 54* | ? / 24.4*

<div id='license'/>

## License

This code uses code from other repositories, but the code I wrote myself is under BSD3.
