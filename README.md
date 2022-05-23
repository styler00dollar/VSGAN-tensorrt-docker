# VSGAN-tensorrt-docker

Using image super resolution models with vapoursynth and speeding them up with TensorRT if possible. This repo is the fastest inference code that you can find. Not all codes can use TensorRT due to various reasons, but I try to add that if it works. Using [NVIDIA/Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) combined with [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN). This repo makes the usage of tiling and ESRGAN models very easy. Models can be found on the [wiki page](https://upscale.wiki/wiki/Model_Database). Further model architectures are planned to be added later on.

Table of contents
=================

<!--ts-->
   * [Usage](#usage)
   * [Deduplicated inference](#deduplicated)
   * [Skipping scenes with scene detection](#skipping)
   * [vs-mlrt (C++ TRT)](#vs-mlrt)
   * [ncnn](#ncnn)
       * [If you have errors installing ncnn whl files with pip](#pip-error)
       * [Rife ncnn](#rife-ncnn)
       * [RealSR / ESRGAN ncnn](#sr-ncnn)
       * [Waifu2x ncnn](#waifu-ncnn)
   * [VFR (variable refresh rate)](#vfr)
   * [mpv](#mpv)
   * [Color transfer](#color)
   * [Benchmarks](#benchmarks)
   * [License](#license)
<!--te-->

-------

Currently working:
- ESRGAN with [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN) and [HolyWu/vs-realesrgan](https://github.com/HolyWu/vs-realesrgan)
- RealESRGAN / RealESERGANVideo with [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN)
- RealESRGAN ncnn with [styler00dollar/realsr-ncnn-vulkan-python](https://github.com/styler00dollar/realsr-ncnn-vulkan-python)
- [Rife4 with HolyWu/vs-rife](https://github.com/HolyWu/vs-rife/)
- Rife ncnn with [styler00dollar/realsr-ncnn-vulkan-python](https://github.com/styler00dollar/realsr-ncnn-vulkan-python)
- [SwinIR with HolyWu/vs-swinir](https://github.com/HolyWu/vs-swinir)
- [Sepconv (enhanced) with sniklaus/revisiting-sepconv](https://github.com/sniklaus/revisiting-sepconv/)
- EGVSR with [Thmen/EGVSR](https://github.com/Thmen/EGVSR) and [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- BasicVSR++ with [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- Waifu2x with [Nlzy/vapoursynth-waifu2x-ncnn-vulkan](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan)
- RealBasicVSR with [ckkelvinchan/RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR)
- RealCUGAN with [bilibili/ailab](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md)
- FILM with [google-research/frame-interpolation](https://github.com/google-research/frame-interpolation)
- TensorRT C++ inference with [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- PAN with [zhaohengyuan1/PAN](https://github.com/zhaohengyuan1/PAN)

Model | ESRGAN | SRVGGNetCompact | Rife | SwinIR | Sepconv | EGVSR | BasicVSR++ | Waifu2x | RealBasicVSR | RealCUGAN | FILM | DPIR | PAN
---  | ------- | --------------- | ---- | ------ | ------- | ----- | ---------- | ------- | ------------ | --------- | ---- | ---- | ---
CUDA | - | [yes](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0) | yes ([rife40](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view), [rife41](https://drive.google.com/file/d/1CPJOzo2CHr8AN3GQCGKOKMVXIdt1RBR1/view)) | [yes](https://github.com/HolyWu/vs-swinir/tree/master/vsswinir) | [yes](http://content.sniklaus.com/resepconv/network-paper.pytorch) | [yes](https://github.com/Thmen/EGVSR/raw/master/pretrained_models/EGVSR_iter420000.pth) | [yes](https://github.com/HolyWu/vs-basicvsrpp/releases/tag/model) | - | [yes](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) | [yes](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD) | [yes](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy) | - | [yes](https://github.com/zhaohengyuan1/PAN/tree/master/experiments/pretrained_models)
TensoRT | yes (torch_tensorrt / C++ TRT) | yes (onnx_tensorrt / C++ TRT) | - | - | - | - | - | yes (C++ TRT) | - | - | - | yes (C++ TRT) | -
ncnn | yes ([realsr ncnn models](https://github.com/nihui/realsr-ncnn-vulkan/tree/master/models)) | yes ([2x](https://files.catbox.moe/u62vpw.tar)) | [yes (all nihui models)](https://github.com/nihui/rife-ncnn-vulkan/tree/master/models) | - | - | - | - | [yes](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan/releases/download/r0.1/models.7z) | - | - | -
onnx | - | yes | - | - | - | - | - | - | - | yes | - | - | -



Algo | CUDA | TensoRT | ncnn | onnx
ESRGAN | - | yes | yes | yes

Some important things:
- Using `Webm` video files is unadvised since it often results in broken results.
- Processing variable framerate (vfr) video is dangerous, but you can try to use fpsnum and fpsden. I would recommend to just render the input video into constant framerate (crf).
- `x264` can be faster than `ffmpeg`, try that out instead.
- `ncnn` does not work with docker. Docker can only support Nvidia GPUs and even if you want to run ncnn with a supported GPU inside docker, you will just get llvmpipe instead of GPU acceleration. If you want ncnn, install dependencies to your own system.
- `rife4` can use PSNR, SSIM, MS_SSIM deduplication. Quick testing showed quite some speed increase.
- Colabs have a weak cpu, you should try `x264` with `--opencl` (A100 does not support NVENC and such).

<div id='usage'/>

## Usage
```bash
# install docker, command for arch
yay -S docker nvidia-docker nvidia-container-toolkit docker-compose

# Build docker
# Put the dockerfile in a directory and run that inside that directory
# You can name it whatever you want, I just applied the same name as my github repo.
docker build -t xpscyho/vsgan_tensorrt:latest .
# If you want to rebuild from scratch or have errors, try to build without cache
# If you still have problems, try to uncomment "RUN apt-get dist-upgrade -y" in the Dockerfile and try again
docker build --no-cache -t xpscyho/vsgan_tensorrt:latest . 
# If you encounter 401 unauthorized error, use this command before running docker build
docker pull nvcr.io/nvidia/tensorrt:21.12-py3

# run the docker with docker-compose
# go into the vsgan folder, inside that folder should be compose.yaml, run this command
# you can adjust folder mounts in the yaml file
docker-compose run --rm vsgan_tensorrt

# you can use it in various ways, ffmpeg example
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv

# av1an is supported too (with svt)
# Warning: Currently frame interpolation does not properly work, but upscaling does
# Torch-TensorRT backend seems to break, C++ TRT seems to work. Either use engine or CUDA.
av1an -e svt-av1 -i inference.py -o output.mkv
```

If docker does not want to start, try this before you use docker:
```bash
# fixing docker errors
systemctl start docker
sudo chmod 666 /var/run/docker.sock
```
Windows is mostly similar, but the path needs to be changed slightly:
```
Example for C://path
docker run --privileged --gpus all -it --rm -v /mnt/c/path:/workspace/tensorrt vsgan_tensorrt:latest
docker run --privileged --gpus all -it --rm -v //c/path:/workspace/tensorrt vsgan_tensorrt:latest
```

If you are confused, here is a Youtube video showing how to use Python API based TensorRT on Windows. That's the easiest way to get my code running, but I would recommend trying to create `.engine` files instead. I wrote instructions for that further down below under [vs-mlrt (C++ TRT)](#vs-mlrt). The difference in speed can be quite big. Look at [benchmarks](#benchmarks) for further details.

[![Tutorial](https://img.youtube.com/vi/B134jvhO8yk/0.jpg)](https://www.youtube.com/watch?v=B134jvhO8yk)

There is also batch processing, just edit and use `main.py` (which calls `inference_batch.py`, edit the file if needed) instead.
```bash
python main.py
```
<div id='deduplicated'/>

## Deduplicated inference
You can delete and duplicate video frames, so you only process non-duplicated frames. These examples can also be viewed inside `inference.py.`
```python
from src.dedup import return_frames
frames_duplicated, frames_duplicating = return_frames(video_path, psnr_value=60)
clip = core.std.DeleteFrames(clip, frames_duplicated)
# place upscaling commands here
clip = core.std.DuplicateFrames(clip, frames_duplicating)
```

<div id='skipping'/>

## Skipping scenes with scene detection
This avoids interpolation when a scene change happens. Create framelist with pyscenedetect and pass that.
```python
from src.scene_detect import find_scenes
skip_framelist = find_scenes(video_path, threshold=30)
clip = RIFE(clip, multi = 2, scale = 1.0, fp16 = True, fastmode = False, ensemble = True, psnr_dedup = False, psnr_value = 70, ssim_dedup = True, ms_ssim_dedup = False, ssim_value = 0.999, skip_framelist=skip_framelist)
```

<div id='vs-mlrt'/>

## vs-mlrt (C++ TRT)
You need to convert onnx models into engines. You need to do that on the same system where you want to do inference. Download onnx models from [here]( https://github.com/AmusementClub/vs-mlrt/releases/download/v7/models.v7.7z) or from [my Github page](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models). You can technically just use any ONNX model you want or convert a pth into onnx with [convert_esrgan_to_onnx.py](https://github.com/styler00dollar/VSGAN-tensorrt-docker/blob/main/convert_esrgan_to_onnx.py). Inside the docker, you do
```
trtexec --fp16 --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT
```
Be aware that DPIR (color) needs 4 channels.
```
trtexec --fp16 --onnx=dpir_drunet_color.onnx --minShapes=input:1x4x8x8 --optShapes=input:1x4x720x1280 --maxShapes=input:1x4x1080x1920 --saveEngine=model.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT
```
and put that engine path into `inference.py`. Only do FP16 if your GPU does support it.

<div id='ncnn'/>

## ncnn
If you want to use ncnn, then you need to set up your own os for this and install dependencies manually. I tried to create a docker, but it isn't working properly. 

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

<div id='rife-ncnn'/>

#### Rife ncnn:
You can install precompiled whl files from [here](https://github.com/styler00dollar/rife-ncnn-vulkan-python/releases/tag/v1a). If you want to compile it, visit [styler00dollar/rife-ncnn-vulkan-python](https://github.com/styler00dollar/rife-ncnn-vulkan-python).
```bash
sudo pacman -S base-devel vulkan-headers vulkan-icd-loader vulkan-devel
pip install [URL for whl]
```

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

#### Waifu2x ncnn:
```python
sudo pacman -S vapoursynth glslang vulkan-icd-loader vulkan-headers

git clone https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan.git
cd vapoursynth-waifu2x-ncnn-vulkan
git submodule update --init --recursive
mkdir build
cd build
cmake ..
cmake --build . -j16
sudo su
make install
exit
```

<div id='vfr'/>

## VFR

**Warning**:Using variable refresh rate video input will result in desync errors. To check if a video is VFR, use this command:
```bash
ffmpeg -i video_Name.mp4 -vf vfrdet -f null -

#If the final line is not zero, then it means it is variable refresh rate. Example:

#[Parsed_vfrdet_0 @ 0x56518fa3f380] VFR:0.400005 (15185/22777) min: 1801 max: 3604)

#To go around this issue, simply convert everything to constant framerate with ffmpeg.

ffmpeg -i video_input.mkv -vsync cfr -crf 10 -c:a copy video_out.mkv

or use my `vfr_to_cfr.py` to process a folder.
```
## Manual instructions
If you don't want to use docker, vapoursynth install commands are [here](https://github.com/styler00dollar/vs-vfi) and a TensorRT example is [here](https://github.com/styler00dollar/Colab-torch2trt/blob/main/Colab-torch2trt.ipynb).

Set the input video path in `inference.py` and access videos with the mounted folder.

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
- The 3090 benches were done with a low powerlimit and throttled the GPU.
- The default is ffmpeg.
- ModifyFrame is depricated. Trying to use FrameEval everywhere and is used by default.
- All ncnn benchmarks are done after [this commit which did improve ncnn performance](https://github.com/Tencent/ncnn/commit/6e19ab26ba82a35c9b7c306bd9519c38ade26bf3).

Compact (2x) | 480p | 720p | 1080p
------  | ---  | ---- | ------
rx470 vs+ncnn (np+no tile+tta off) | 2.7 | 1.6 | 0.6
1070ti vs+ncnn (np+no tile+tta off) | 4.2 | 2 | 0.9
1070ti TensorRT8 docker 2x (ONNX-TRT+FrameEval) | 12 | 6.1 | 2.8
1070ti TensorRT8 docker 2x (C++ TRT+FrameEval+num_streams=6) | 14 | 6.7 | 3
3060ti TensorRT8 docker 2x (ONNX-TRT+FrameEval) | 19 | 7.1 | 3.2
3060ti TensorRT8 docker 2x (C++ TRT+FrameEval+num_streams=5) | 47.93 | 15.97 | 7.83
3060ti VSGAN 2x | 9.7 | 3.6 | 1.77
3060ti ncnn (Windows binary) 2x | 7 | 4.2 | 1.2
3060ti Joey 2x | 2.24 | 0.87 | 0.36
3070 TensorRT8 docker 2x (ONNX-TRT+FrameEval) | 20 | 7.55 | 3.36
3090 TensorRT8 docker 2x (ONNX-TRT+FrameEval) | ? | ? | 6.7
3090 TensorRT8 docker 2x (C++ TRT+FrameEval+num_streams=22) | 61 | ? | 14
V100 (Colab) (vs+CUDA) | 8.4 | 3.8 | 1.6
V100 (Colab) (vs+TensorRT8+ONNX-TRT+FrameEval) | 8.3 | 3.8 | 1.7
V100 (Colab High RAM) (vs+CUDA+FrameEval) | 29 | 13 | 6
V100 (Colab High RAM) (vs+TensorRT7+ONNX-TRT+FrameEval) | 21 | 12 | 5.5
V100 (Colab High RAM) (vs+TensorRT8+ONNX-TRT+FrameEval) | 21 | 12 | 5.5
A100 (Colab) (vs+CUDA+FrameEval) | 40 | 19 | 8.5
A100 (Colab) (vs+TensorRT8+ONNX-TRT+FrameEval) | 44 | 21 | 9.5
A100 (Colab) (vs+TensorRT8+C++ TRT+ffmpeg+FrameEval+num_streams=50) | 52.72 | 24.37 | 11.84
A100 (Colab) (vs+TensorRT8) (C++ TRT+x264 (--opencl)+FrameEval+num_streams=50) | 57.16 | 26.25 | 12.42
A100 (Colab) (vs+onnx+FrameEval) | 26 | 12 | 4.9
A100 (Colab) (vs+quantized onnx+FrameEval) | 26 | 12 | 5.7
A100 (Colab) (jpg+CUDA) | 28.2 (9 Threads) | 28.2 (7 Threads) | 9.96 (4 Threads)


Compact (4x) | 480p | 720p | 1080p
------  | ---  | ---- | ------
1070ti TensorRT8 docker (ONNX-TensorRT+FrameEval) | 11 | 5.6 | X
3060ti TensorRT8 docker (ONNX-TensorRT+FrameEval) | 16 | 6.1 | 2.7
3060ti TensorRT8 docker 2x (C++ TRT+FrameEval+num_streams=5) | 29.78 | 11 | 5.24
3060ti VSGAN 4x | 7.2 | 3 | 1.3
3060ti ncnn (Windows binary) 4x | 3.72 | 0.85 | 0.53
3060ti Joey 4x | 0.65 | 0.25 | 0.11
A100 (Colab) (vs+CUDA+FrameEval) | 12 | 5.6 | 2.9
A100 (Colab) (jpg+CUDA) | ? | ?| 3 (4 Threads)

ESRGAN 4x (64mb) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
1070ti TensorRT8 docker (Torch-TensorRT+ffmpeg+FrameEval) | 0.5 | 0.2 | >0.1
3060ti TensorRT8 docker (Torch-TensorRT+ffmpeg+FrameEval) | 2 | 0.7 | 0.29
3060ti Cupscale (Pytorch) | 0.41 | 0.13 | 0.044
3060ti Cupscale (ncnn) | 0.27 | 0.1 | 0.04
3060ti Joey | 0.41 | 0.095 | 0.043
V100 (Colab) (Torch-TensorRT+ffmpeg+FrameEval) | 1.8 | 0.8 | ?
V100 (Colab High VRAM) (C++ TensorRT+x264 (--opencl)+FrameEval+no tiling) | 2.46 | OOM (OpenCL) | OOM (OpenCL)
V100 (Colab High VRAM) (C++ TensorRT+x264+FrameEval+no tiling) | 2.49 | 1.14 | 0.47
A100 (Colab) (Torch-TensorRT+ffmpeg+FrameEval) | 5.6 | 2.6 | 1.1

RealESRGAN (2x) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=1) | 8.14 | 3.12 | 1.4

RealESRGAN (4x) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
3060ti (vs+TensorRT8+ffmpeg+C++ TRT+num_streams=2) | 6.8 | 1.7 | 0.75
V100 (Colab High RAM) (vs+TensorRT8+x264 (--opencl)+C++ TRT+num_streams=1+no tiling) | 6.82 | 3.15 | OOM (OpenCL) 
V100 (Colab High RAM) (vs+TensorRT8+x264+C++ TRT+num_streams=1+no tiling) | ? | ? | 1.39
A100 (vs+TensorRT8+x264 (--opencl)+C++ TRT+num_streams=3+no tiling) | 14.65 | 6.74 | 2.76

Rife4+vs (fastmode False, ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (vs+ffmpeg+ModifyFrame) | 61 | 30 | 15
3060ti (vs+ffmpeg+ModifyFrame) | 89 | 45 | 24

Rife4+vs (fastmode False, ensemble True) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (vs+ffmpeg+ModifyFrame) | 27 | 13 | 9.6
3060ti (vs+ffmpeg+ModifyFrame) | ? | 36 | 20 |
3090 (vs+ffmpeg+ModifyFrame) | ? | 69.6 | 35 | 
V100 (Colab) (vs+ffmpeg+ModifyFrame) | 30 | 16 | 7.3
V100 (Colab High RAM) (vs+x264+ModifyFrame) | 48.5 | 33 | 19.2
V100 (Colab High RAM) (vs+x264+FrameEval) | 48.2 | 35.5 | 20.6
V100 (Colab High RAM) (vs+x265+FrameEval) | 15.2 | 9.7 | 4.6
A100 (Colab) (vs+CUDA+ffmpeg+ModifyFrame) | 54 | 39 | 23
A100 (Colab) (jpg+CUDA+ffmpeg+ModifyFrame) | ? | ? | 19.92 (14 Threads)

Rife4+vs (fastmode True, ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (TensorRT8+ffmpeg+ModifyFrame) | 62 | 31 | 14
3060ti (TensorRT8+ffmpeg+ModifyFrame) | 135 | 66 | 33 |
3090 (TensorRT8+ffmpeg+ModifyFrame) | ? | 119 | 58 | 
V100 (Colab) (TensorRT8+ffmpeg+ModifyFrame) | 34 | 17 | 7.6
A100 (Colab) (TensorRT8+ffmpeg+ModifyFrame) | 92 | 56 | 29

Rife4+vs (fastmode True, ensemble True) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (TensorRT8+ffmpeg+ModifyFrame) | 41 | 20 | 9.8 
3060ti (TensorRT8+ffmpeg+ModifyFrame) | 86 | 49 | 24 | 
3090 (TensorRT8+ffmpeg+ModifyFrame) | ? | 90.3 | 45

Rife4+vs ncnn | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
rx470 (ffmpeg+FrameEval) | 11 | 6.2 | 3.4
1070ti (ffmpeg+FrameEval) | 34 | 16 | 7.5

EGVSR | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
1070ti | 3.1 | OOM | OOM
V100 (Colab) | 2 | ? | ?
A100 (Colab) | 5.7 | 2.3 | 1.4

RealBasicVSR | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
1070ti | 0.3 | OOM | OOM
A100 (Colab) | 1.2 | ? | ?

Sepconv | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
V100 (Colab) | 22 | 11 | 4.9

CAIN (2 groups) | 480p | 720p | 1080p 
-----------  | ---- | ---- | ----
A100 (Colab) | 76 | 47 | 25

cugan 2x | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
V100 (Colab) (vs+CUDA) | 7 | 3.1 | ?
V100 (Colab High RAM) (vs+CUDA) | 21 | 9.7 | 4

cugan 4x | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
3090 | 26 | ? | ?

FILM | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
V100 (Colab High RAM) (vs+CUDA) | 9.8 | 4.7 | 2.1

## Combined Benchmarks

Rife4 (fastmode False, ensemble True) + Compact 2x | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (ONNX-TensorRT8+ffmpeg+ModifyFrame) | 9.3 | 4.6 | 2.2
1070ti (C++ TensorRT8+ffmpeg+ModifyFrame) | ? | ? | 2.7
V100 (Colab High RAM) (vs+CUDA+ffmpeg+ModifyFrame) | ? | ? | 5.1
V100 (Colab High RAM) (vs+CUDA+x264+ModifyFrame) | ? | ? | 5.2
V100 (Colab High RAM) (vs+CUDA+x264+FrameEval) | ? | ? | 5.1
V100 (Colab High RAM) (vs+ONNX-TensorRT8+x264+ModifyFrame) (rife fp16=False) | ? | ? | 4.2
A100 (Colab) (vs+CUDA+ffmpeg+ModifyFrame) | 23 | 13 | 6.6
A100 (Colab) (vs+ONNX-TensorRT8+ffmpeg+ModifyFrame) (rife fp16=False) | 27 | 15 | 7.4
A100 (Colab) (vs+ONNX-TensorRT8+ffmpeg+ModifyFrame) (rife fp16=False) | 27 | 15 | 7.4
A100 (Colab) (vs+C++ TensorRT8+ffmpeg+FrameEval) (num_streams=49) | ~29 | ~18 | 9.96
A100 (Colab) (vs+C++ TensorRT8+x264 (--opencl)+FrameEval) (num_streams=49) | 30.10 | 19.81 | 10.6

Rife4 (fastmode False, ensemble True) + RealESRGAN (4x) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
A100 (vs+TensorRT8+x264 (--opencl)+C++ TRT+num_streams=2+no tiling) | 14.46 | 7.39 | 3.18

Rife4 + cugan 2x | 480p | 720p | 1080p 
-------- | ---- | ---- | ----
A100 (vs+CUDA+ffmpeg+FrameEval) | 19 | 10 | 5

<div id='license'/>

## License

styler00dollar: This code uses code from other repositories, but the code I wrote myself is under BSD3.

xpscyho:  Mine too! for what that's worth