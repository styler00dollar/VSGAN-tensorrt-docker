# VSGAN-tensorrt-docker

Using image super resolution models with vapoursynth and speeding them up with TensorRT. Using [NVIDIA/Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) combined with [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN). This repo makes the usage of tiling and ESRGAN models very easy. Models can be found on the [wiki page](https://upscale.wiki/wiki/Model_Database). Further model architectures are planned to be added later on.

Currently working:
- ESRGAN with [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN) and [HolyWu/vs-realesrgan](https://github.com/HolyWu/vs-realesrgan)
- RealESRGAN / RealESERGANVideo with [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN)
- RealESRGAN ncnn with [styler00dollar/realsr-ncnn-vulkan-python](https://github.com/styler00dollar/realsr-ncnn-vulkan-python)
- [Rife4 with HolyWu/vs-rife](https://github.com/HolyWu/vs-rife/)
- Rife ncnn with [DaGooseYT/VapourSynth-RIFE-ncnn-Vulkan](https://github.com/DaGooseYT/VapourSynth-RIFE-ncnn-Vulkan)
- [SwinIR with HolyWu/vs-swinir](https://github.com/HolyWu/vs-swinir)
- [Sepconv (enhanced) with sniklaus/revisiting-sepconv](https://github.com/sniklaus/revisiting-sepconv/)
- EGVSR with [Thmen/EGVSR](https://github.com/Thmen/EGVSR) and [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- BasicVSR++ with [HolyWu/vs-basicvsrpp](https://github.com/HolyWu/vs-basicvsrpp)
- Waifu2x with [Nlzy/vapoursynth-waifu2x-ncnn-vulkan](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan)
- RealBasicVSR with [ckkelvinchan/RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR)

Model | ESRGAN | SRVGGNetCompact | Rife | SwinIR | Sepconv | EGVSR | BasicVSR++ | Waifu2x | RealBasicVSR
---  | ------- | --------------- | ---- | ------ | ------- | ----- | ---------- | ------- | ------------
CUDA | - | [yes](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0) | yes ([rife4](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view)) | [yes](https://github.com/HolyWu/vs-swinir/tree/master/vsswinir) | [yes](http://content.sniklaus.com/resepconv/network-paper.pytorch) | [yes](https://github.com/Thmen/EGVSR/raw/master/pretrained_models/EGVSR_iter420000.pth) | [yes](https://github.com/HolyWu/vs-basicvsrpp/releases/tag/model) | - | [yes](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view)
TensoRT | yes (torch_tensorrt) | yes (onnx_tensorrt) | - | - | - | - | - | - | -
ncnn | yes ([realsr ncnn models](https://github.com/nihui/realsr-ncnn-vulkan/tree/master/models)) | yes ([2x](https://files.catbox.moe/u62vpw.tar)) | yes ([rife3.1, 3.0, 2.4, 2, anime](https://github.com/DaGooseYT/VapourSynth-RIFE-ncnn-Vulkan/tree/master/models)) | - | - | - | - | [yes](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan/releases/download/r0.1/models.7z) | -

## Usage
```bash
# install docker, command for arch
yay -S docker nvidia-docker nvidia-container-toolkit
# Put the dockerfile in a directory and run that inside that directory
docker build -t vsgan_tensorrt:latest .
# run the docker
# the folderpath before ":" will be mounted in the path which follows afterwards
# contents of the vsgan folder should appear inside /workspace/tensorrt
docker run --privileged --gpus all -it --rm -v /home/vsgan_path/:/workspace/tensorrt vsgan_tensorrt:latest
# you can use it in various ways, ffmpeg example
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv
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
There is also batch processing, just edit and use `main.py` (which calls `inference_batch.py`, edit the file if needed) instead.
```bash
python main.py
```
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
#### Rife ncnn:
```bash
sudo pacman -S base-devel cmake vulkan-headers vulkan-icd-loader python
pip install meson ninja

git clone https://github.com/DaGooseYT/VapourSynth-RIFE-ncnn-Vulkan
cd VapourSynth-RIFE-ncnn-Vulkan && git submodule update --init --recursive --depth 1 && meson build && ninja -C build install
```
#### RealSR / ESRGAN ncnn:
```bash
sudo pacman -S base-devel cmake vulkan-headers vulkan-icd-loader swig python
pip install cmake-build-extension numpy -U

# dont use conda, CXX errors in manjaro otherwise
conda deactivate
git clone https://github.com/styler00dollar/realsr-ncnn-vulkan-python
cd realsr-ncnn-vulkan-python/realsr_ncnn_vulkan_python/realsr-ncnn-vulkan/
git submodule update --init --recursive
cd src

# There are 2 CMakeLists.txt
# Make sure that prelu is set to ON, otherwise the compact model wont work
# option(WITH_LAYER_prelu "" ON)

# comment this line in the realsr.cpp file
# fprintf(stderr, "%.2f%%\n", (float)(yi * xtiles + xi) / (ytiles * xtiles) * 100);

# if you dont want the 2 default pth files in your whl / install,
# comment the lines with say "models" in CMakeLists.txt

# replace realsr_ncnn_vulkan_without_PIL.py with realsr_ncnn_vulkan.py

cmake -B build .
cd build
make -j16
sudo su
make install
exit
cd .. && cd .. && cd .. && cd ..
python setup.py install --user
```
Any ESRGAN model will work with this, when you have the fitting param file. Make sure the input is called "data" and output is "output".

If you want to convert a normal pth to ncnn, you need to do `pth->onnx->ncnn(bin/param)`. For the first step you can use `torch.onnx` and for the second one you can use [this website](https://convertmodel.com/).

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

## VFR
**Warning**: Using variable refresh rate video input will result in desync errors. To check if a video is do
```bash
ffmpeg -i video_Name.mp4 -vf vfrdet -f null -
```
and look at the final line. If it is not zero, then it means it is variable refresh rate. Example:
```bash
[Parsed_vfrdet_0 @ 0x56518fa3f380] VFR:0.400005 (15185/22777) min: 1801 max: 3604)
```
To go around this issue, simply convert everything to constant framerate with ffmpeg.
```bash
ffmpeg -i video_input.mkv -vsync cfr -crf 10 -c:a copy video_out.mkv
```
or use my `vfr_to_cfr.py` to process a folder.
## Manual instructions
If you don't want to use docker, vapoursynth install commands are [here](https://github.com/styler00dollar/vs-vfi) and a TensorRT example is [here](https://github.com/styler00dollar/Colab-torch2trt/blob/main/Colab-torch2trt.ipynb).

Set the input video path in `inference.py` and access videos with the mounted folder.
## mpv
It is also possible to directly pipe the video into mpv, but you most likely wont be able to archive realtime speed. Change the mounted folder path to your own videofolder and use the mpv dockerfile instead. If you use a very efficient model, it may be possible on a very good GPU. Only tested in Manjaro. 
```bash
yay -S pulseaudio

# i am not sure if it is needed, but go into pulseaudio settings and check "make pulseaudio network audio devices discoverable in the local network" and reboot

# start docker
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
## Benchmarks

Warning: The 3090 benches were done with a low powerlimit and throttled the GPU.

Compact (2x) | 480p | 720p | 1080p
------  | ---  | ---- | ------
rx470 vs+ncnn (PIL+no tile+tta off) 2x | 2.8 | 1.3 | 0.5
rx470 vs+ncnn (np+no tile+tta off) 2x | 2.8 | 1.3 | 0.5
rx470 vs+ncnn (np+auto tile+tta off) 2x | 2.5 | 1.1 | 0.4
rx470 vs+ncnn (np+no tile+tta on) 2x | 0.4 | 0.2 | X
1070ti vs+ncnn (np+no tile+tta off) | 3.8 | 1.7 | 0.7
1070ti TensorRT8 docker 2x | 12 | 6.1 | 2.8
3060ti TensorRT8 docker 2x | 19 | 7.1 | 3.2
3060ti VSGAN 2x | 9.7 | 3.6 | 1.77
3060ti ncnn (Windows binary) 2x | 7 | 4.2 | 1.2
3060ti Joey 2x | 2.24 | 0.87 | 0.36
3070 TensorRT8 docker 2x | 20 | 7.55 | 3.36
3090 TensorRT8 docker 2x | ? | ? | 6.7
V100 (Colab) (vs+CUDA) | 6.9 | 3.2 | 1.4
V100 (Colab High RAM) (vs+CUDA) | 31 | 15 | 6.4
V100 (Colab High RAM) (vs+TensorRT7) | 21 | 12 | 5.5
A100 (Colab) (vs+CUDA) | 40 | 19 | 8.5
A100 (Colab) (vs+onnx) | 26 | 12 | 4.9
A100 (Colab) (vs+quantized onnx) | 26 | 12 | 5.7
A100 (Colab) (jpg+CUDA) | 28.2 (9 Threads) | 28.2 (7 Threads) | 9.96 (4 Threads)

Compact (4x) | 480p | 720p | 1080p
------  | ---  | ---- | ------
1070ti TensorRT8 docker 4x | 11 | 5.6 | X
3060ti TensorRT8 docker 4x | 16 | 6.1 | 2.7
3060ti VSGAN 4x | 7.2 | 3 | 1.3
3060ti ncnn (Windows binary) 4x | 3.72 | 0.85 | 0.53
3060ti Joey 4x | 0.65 | 0.25 | 0.11
A100 (Colab) (vs+CUDA) | 12 | 5.6 | 2.9
A100 (Colab) (jpg+CUDA) | ? | ?| 3 (4 Threads)

ESRGAN (64mb) | 480p | 720p | 1080p
------------  | ---  | ---- | ------
1070ti TensorRT8 docker 4x | 0.5 | 0.2 | >0.1
3060ti TensorRT8 docker 4x | 2 | 0.7 | 0.29
3060ti Cupscale (Pytorch) 4x | 0.41 | 0.13 | 0.044
3060ti Cupscale (ncnn) 4x | 0.27 | 0.1 | 0.04
3060ti Joey 4x | 0.41 | 0.095 | 0.043
V100 TensoRT8 (Colab) | 1.8 | 0.8 | ?
A100 TensoRT8 (Colab) | 5.6 | 2.6 | 1.1

Rife4+vs (fastmode False, ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti | 61 | 30 | 15
3060ti | 89 | 45 | 24 | 

Rife4+vs (fastmode False, ensemble True) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti | 27 | 13 | 9.6
3060ti | ? | 36 | 20 |
3090 | ? | 69.6 | 35 | 
V100 (Colab) | 30 | 16 | 7.3
A100 (Colab) (vs+CUDA) | 54 | 39 | 23
A100 (Colab) (jpg+CUDA) | ? | ? | 19.92 (14 Threads)

Rife4+vs (fastmode True, ensemble False) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti | 62 | 31 | 14
3060ti | 135 | 66 | 33 |
3090 | ? | 119 | 58 | 
V100 (Colab) | 34 | 17 | 7.6
A100 (Colab) | 92 | 56 | 29

Rife4+vs (fastmode True, ensemble True) | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti | 41 | 20 | 9.8 
3060ti | 86 | 49 | 24 | 
3090 | ? | 90.3 | 45

Rife4+vs (fastmode False, ensemble True) + Compact 2x | 480p | 720p | 1080p 
---  | -------  | ------- | ------- 
1070ti (TensorRT8) | 9.3 | 4.6 | 2.2
V100 (Colab High RAM) (vs+CUDA) | 20 | 11 | 5.5
A100 (Colab) (vs+CUDA) | 23 | 13 | 6.6
A100 (Colab) (jpg+CUDA)  | ? | ? | 6.66 (theoretical)

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
