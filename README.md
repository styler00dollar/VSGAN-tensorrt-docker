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

Model | ESRGAN | SRVGGNetCompact | Rife | SwinIR | Sepconv | EGVSR | BasicVSR++ | Waifu2x
---  | ------- | --------------- | ---- | ------ | ------- | ----- | ---------- | -------
CUDA | - | - | yes ([rife4](https://drive.google.com/file/d/1mUK9iON6Es14oK46-cCflRoPTeGiI_A9/view)) | [yes](https://github.com/HolyWu/vs-swinir/tree/master/vsswinir) | [yes](http://content.sniklaus.com/resepconv/network-paper.pytorch) | [yes](https://github.com/Thmen/EGVSR/raw/master/pretrained_models/EGVSR_iter420000.pth) | [yes](https://github.com/HolyWu/vs-basicvsrpp/releases/tag/model) | -
TensoRT | yes (torch_tensorrt) | yes (onnx_tensorrt) | - | - | - | - | - | -
ncnn | yes ([realsr ncnn models](https://github.com/nihui/realsr-ncnn-vulkan/tree/master/models)) | yes ([2x](https://files.catbox.moe/u62vpw.tar)) | yes ([rife3.1, 3.0, 2.4, 2, anime](https://github.com/DaGooseYT/VapourSynth-RIFE-ncnn-Vulkan/tree/master/models)) | - | - | - | - | [yes](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan/releases/download/r0.1/models.7z)

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
Rife ncnn:
```bash
sudo pacman -S base-devel cmake vulkan-headers vulkan-icd-loader python
pip install meson ninja

git clone https://github.com/DaGooseYT/VapourSynth-RIFE-ncnn-Vulkan
cd VapourSynth-RIFE-ncnn-Vulkan && git submodule update --init --recursive --depth 1 && meson build && ninja -C build install
```
RealSR / ESRGAN ncnn:
```bash
sudo pacman -S base-devel cmake vulkan-headers vulkan-icd-loader swig python

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

Waifu2x ncnn
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
