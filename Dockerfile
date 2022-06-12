# https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
FROM nvcr.io/nvidia/tensorrt:22.04-py3
ARG DEBIAN_FRONTEND=noninteractive
# if you have 404 problems when you build the docker, try to run the upgrade
#RUN apt-get dist-upgrade -y
RUN apt-get -y update

# installing vapoursynth and torch
RUN apt install p7zip-full x264 ffmpeg autoconf libtool yasm python3.9 python3.9-venv python3.9-dev ffmsindex libffms2-4 libffms2-dev -y && \
    git clone https://github.com/sekrit-twc/zimg.git && cd zimg && ./autogen.sh && ./configure && make -j4 && make install && cd .. && rm -rf zimg && \
    pip install Cython && wget https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R57.zip && \
    7z x R57.zip && cd vapoursynth-R57 && ./autogen.sh && ./configure && make && make install && cd .. && ldconfig && \
    ln -s /usr/local/lib/python3.9/site-packages/vapoursynth.so /usr/lib/python3.9/lib-dynload/vapoursynth.so && \
    pip install scipy mmedit vapoursynth meson ninja numba numpy scenedetect kornia opencv-python onnx onnxruntime onnxruntime-gpu cupy-cuda115 pytorch-msssim \
        https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp38-cp38-linux_x86_64.whl \
        https://download.pytorch.org/whl/cpu/torchvision-0.12.0%2Bcpu-cp38-cp38-linux_x86_64.whl \
        https://github.com/pytorch/TensorRT/releases/download/v1.1.0/torch_tensorrt-1.1.0-cp38-cp38-linux_x86_64.whl \
        mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html && \
    # not deleting vapoursynth-R57 since vs-mlrt needs it
    rm -rf R57.zip zimg && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y && pip3 cache purge

# upgrading ffmpeg manually (ffmpeg 20220422 from https://johnvansickle.com/ffmpeg/)
RUN wget https://files.catbox.moe/nm1sfp -O ffmpeg && \
    chmod +x ./ffmpeg && mv ffmpeg /usr/bin/ffmpeg

# installing tensorflow because of FILM
RUN pip install tensorflow tensorflow-gpu tensorflow_addons gin-config -U && pip3 cache purge

# installing onnx tensorrt with a workaround, error with import otherwise
# https://github.com/onnx/onnx-tensorrt/issues/643
RUN git clone https://github.com/onnx/onnx-tensorrt.git && \
    cd onnx-tensorrt && \
    cp -r onnx_tensorrt /usr/local/lib/python3.8/dist-packages && \
    cd .. && rm -rf onnx-tensorrt

# downloading models
RUN wget https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/RealESRGANv2-animevideo-xsx2.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/RealESRGANv2-animevideo-xsx4.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/RealESRGAN_x4plus_anime_6B.pth \
# fatal anime
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/4x_fatal_Anime_500000_G.pth \
# rvp1
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rvpV1_105661_G.pt \
# sepconv
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sepconv.pth \
# EGVSR
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/EGVSR_iter420000.pth \
# rife4 (fixed)
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife40.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife41.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/sudo_rife4_269.662_testV1_scale1.pth \
# RealBasicVSR_x4
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/RealBasicVSR_x4.pth \
# cugan models
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up2x-latest-conservative.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up2x-latest-denoise1x.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up2x-latest-denoise2x.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up2x-latest-denoise3x.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up2x-latest-no-denoise.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up3x-latest-conservative.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up3x-latest-denoise3x.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up3x-latest-no-denoise.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up4x-latest-conservative.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up4x-latest-denoise3x.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/up4x-latest-no-denoise.pth \
# IFRNet
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/IFRNet_S_Vimeo90K.pth \
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/IFRNet_L_Vimeo90K.pth \
# film
    https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/FILM.tar.gz && \
    tar -zxvf FILM.tar.gz && rm -rf FILM.tar.gz
# RealBasicVSR_x4 will download this
RUN wget "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" -P /root/.cache/torch/hub/checkpoints/

# vs plugings from others
# https://github.com/HolyWu/vs-swinir
# https://github.com/HolyWu/vs-basicvsrpp
RUN pip install --upgrade vsswinir && python -m vsswinir && \
    pip install --upgrade vsbasicvsrpp && python -m vsbasicvsrpp && \
    pip3 cache purge

# vs-mlrt
# upgrading cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.0-rc1/cmake-3.23.0-rc1-linux-x86_64.sh && \
    chmod +x cmake-3.23.0-rc1-linux-x86_64.sh && sh cmake-3.23.0-rc1-linux-x86_64.sh --skip-license && \
    cp /workspace/bin/cmake /usr/bin/cmake && cp /workspace/bin/cmake /usr/lib/x86_64-linux-gnu/cmake && \
    cp /workspace/bin/cmake /usr/local/bin/cmake && cp -r /workspace/share/cmake-3.23 /usr/local/share/ && \
# upgrading g++
    apt install build-essential manpages-dev software-properties-common -y && add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt update -y && apt install gcc-11 g++-11 -y && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
# compiling
    git clone https://github.com/AmusementClub/vs-mlrt /workspace/vs-mlrt && cd /workspace/vs-mlrt/vstrt && mkdir build && \
    cd build && cmake .. -DVAPOURSYNTH_INCLUDE_DIRECTORY=/workspace/vapoursynth-R57/include && make && make install && \
    cd .. && rm -rf cmake-3.23.0-rc1-linux-x86_64.sh zimg vapoursynth-R57

# x265
RUN git clone https://github.com/AmusementClub/x265 /workspace/x265 && cd /workspace/x265/source/ && mkdir build && cd build && \
    cmake .. -DNATIVE_BUILD=ON -DSTATIC_LINK_CRT=ON -DENABLE_AVISYNTH=OFF && make && make install && \
    cp /workspace/x265/source/build/x265 /usr/bin/x265 && \
    cp /workspace/x265/source/build/x265 /usr/local/bin/x265 && \
    cd .. && rm -rf x265

# descale
RUN git clone https://github.com/Irrational-Encoding-Wizardry/descale && cd descale && meson build && ninja -C build && ninja -C build install && \
    cd .. && rm -rf descale

# mpv
RUN apt install mpv -y && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes pulseaudio-utils && \
    apt-get install -y pulseaudio && apt-get install pulseaudio libpulse-dev osspd -y && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# av1an
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    source $HOME/.cargo/env && \
    apt install clang-12 nasm libavutil-dev libavformat-dev libavfilter-dev -y && \
    git clone https://github.com/master-of-zen/Av1an && \
    cd Av1an && cargo build --release --features ffmpeg_static && \
    mv /workspace/Av1an/target/release/av1an /usr/bin && \
    cd /workspace && rm -rf Av1an && apt-get autoremove -y && apt-get clean
# svt
RUN git clone https://github.com/AOMediaCodec/SVT-AV1 && cd SVT-AV1/Build/linux/ && sh build.sh release && \
    cd /workspace/SVT-AV1/Bin/Release/ && chmod +x ./SvtAv1EncApp && mv SvtAv1EncApp /usr/bin && \
    mv libSvtAv1Enc.so.1.1.0 /usr/local/lib && mv libSvtAv1Enc.so.1 /usr/local/lib && mv libSvtAv1Enc.so /usr/local/lib && \
    cd /workspace && rm -rf SVT-AV1

# pycuda and numpy hotfix
RUN pip install pycuda numpy==1.21 --force-reinstall && pip3 cache purge