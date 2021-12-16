# VSGAN-tensorrt-docker

Using image super resolution models with vapoursynth and speeding them up with TensorRT. Using [NVIDIA/Torch-TensorRT](https://github.com/NVIDIA/Torch-TensorRT) combined with [rlaphoenix/VSGAN](https://github.com/rlaphoenix/VSGAN). This repo makes the usage of tiling and ESRGAN models very easy. Models can be found on the [wiki page](https://upscale.wiki/wiki/Model_Database). Further model architectures are planned to be added later on.

Currently working:
- ESRGAN
- RealESRGAN (adjust model load manually in `inference.py`, settings wont be adjusted automatically currently)

Usage:
```
# install docker, command for arch
yay -S docker nvidia-docker nvidia-container-toolkit
# Put the dockerfile in a directory and run that inside that directory
docker build -t vsgan_tensorrt:latest .
# run with a mounted folder
docker run --privileged --gpus all -it --rm -v /home/Desktop/tensorrt:/workspace/tensorrt vsgan_tensorrt:latest
# you can use it in various ways, ffmpeg example
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv
```

If docker does not want to start, try this before you use docker:
```
# fixing docker errors
systemctl start docker
sudo chmod 666 /var/run/docker.sock
```
Windows is mostly similar, but the path needs to be changed slightly:
```
Example for C://path
docker run --privileged --gpus all -it --rm -v //c/path:/workspace/tensorrt vsgan_tensorrt:latest
```

If you don't want to use docker, vapoursynth install commands are [here](https://github.com/styler00dollar/vs-vfi) and a TensorRT example is [here](https://github.com/styler00dollar/Colab-torch2trt/blob/main/Colab-torch2trt.ipynb).

Set the input video path in `inference.py` and access videos with the mounted folder.

It is also possible to directly pipe the video into mpv, but you most likely wont be able to archive realtime speed. Change the mounted folder path to your own videofolder and use the mpv dockerfile instead. If you use a very efficient model, it may be possible on a very good GPU. Only tested in Manjaro. 
```
yay -S pulseaudio

# i am not sure if it is needed, but go into pulseaudio settings and check "make pulseaudio network audio devices discoverable in the local network" and reboot

# start docker
docker run --rm -i -t \
    --network host \
    -e DISPLAY \
    -v /home/Schreibtisch/test/:/home/mpv/media \
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
```
