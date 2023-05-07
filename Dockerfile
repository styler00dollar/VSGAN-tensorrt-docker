############################
# Vulkan
#https://github.com/bitnimble/docker-vulkan/blob/master/docker/Dockerfile.ubuntu20.04
#https://gitlab.com/nvidia/container-images/vulkan/-/blob/ubuntu16.04/Dockerfile
############################
FROM ubuntu:22.04 as vulkan-khronos

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  pkg-config \
  git \
  libegl1-mesa-dev \
  libwayland-dev \
  libx11-xcb-dev \
  libxkbcommon-dev \
  libxrandr-dev \
  python3 \
  python3-distutils \
  wget && \
  rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.0-rc4/cmake-3.26.0-rc4-linux-x86_64.sh \
  -q -O /tmp/cmake-install.sh && \
  chmod u+x /tmp/cmake-install.sh && \
  mkdir /usr/bin/cmake && \
  /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake && \
  rm /tmp/cmake-install.sh
ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN ln -s /usr/bin/python3 /usr/bin/python && \
  git clone https://github.com/KhronosGroup/Vulkan-ValidationLayers.git /opt/vulkan && \
  cd /opt/vulkan && \
  mkdir build && cd build && ../scripts/update_deps.py && \
  cmake -C helper.cmake -DCMAKE_BUILD_TYPE=Release .. && \
  cmake --build . --config Release -- -j$(nproc) && make install && ldconfig && \
  mkdir -p /usr/local/include/vulkan && cp -r Vulkan-Headers/build/install/include/vulkan/* /usr/local/include/vulkan && \
  cp -r Vulkan-Headers/include/* /usr/local/include/vulkan && \
  mkdir -p /usr/local/share/vulkan/registry && \
  cp -r Vulkan-Headers/build/install/share/vulkan/registry/* /usr/local/share/vulkan/registry && \
  git clone https://github.com/KhronosGroup/Vulkan-Loader /opt/vulkan-loader && \
  cd /opt/vulkan-loader && \
  mkdir build && cd build && ../scripts/update_deps.py && \
  cmake -C helper.cmake -DCMAKE_BUILD_TYPE=Release .. && \
  cmake --build . --config Release -- -j$(nproc) && make install && ldconfig && \
  mkdir -p /usr/local/lib && cp -a loader/*.so* /usr/local/lib && \
  rm -rf /opt/vulkan && rm -rf /opt/vulkan-loader

############################
# FFMPEG
############################
FROM archlinux as ffmpeg-arch
RUN --mount=type=cache,sharing=locked,target=/var/cache/pacman \
  pacman -Syu --noconfirm --needed base base-devel cuda git
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ARG user=makepkg
RUN useradd --system --create-home $user && \
  echo "$user ALL=(ALL:ALL) NOPASSWD:ALL" >/etc/sudoers.d/$user
USER $user
WORKDIR /home/$user
RUN git clone https://aur.archlinux.org/yay.git && \
  cd yay && \
  makepkg -sri --needed --noconfirm && \
  cd && \
  rm -rf .cache yay

RUN yay -Syu && yay -S rust-nightly-bin tcl nasm cmake jq libtool wget fribidi fontconfig libsoxr-git meson pod2man nvidia-utils base-devel python310 --noconfirm --ask 4
USER root

RUN mkdir -p "/home/makepkg/python310"
RUN wget https://github.com/python/cpython/archive/refs/tags/v3.10.10.tar.gz && tar xf v3.10.10.tar.gz && cd cpython-3.10.10 && \
  mkdir debug && cd debug && ../configure --enable-optimizations --disable-shared --prefix="/home/makepkg/python310" && make -j$(nproc) && make install && \
  /home/makepkg/python310/bin/python3.10 -m ensurepip --upgrade
RUN cp /home/makepkg/python310/bin/python3.10 /usr/bin/python
ENV PYTHONPATH /home/makepkg/python310/bin/
ENV PATH "/home/makepkg/python310/bin/:$PATH"

RUN pip3 install Cython meson

ENV PATH "$PATH:/opt/cuda/bin/nvcc"
ENV PATH "$PATH:/opt/cuda/bin"
ENV LD_LIBRARY_PATH "/opt/cuda/lib64"

# -O3 makes sure we compile with optimization. setting CFLAGS/CXXFLAGS seems to override
# default automake cflags.
# -static-libgcc is needed to make gcc not include gcc_s as "as-needed" shared library which
# cmake will include as a implicit library.
# other options to get hardened build (same as ffmpeg hardened)
ARG CFLAGS="-O3 -static-libgcc -fno-strict-overflow -fstack-protector-all -fPIE"
ARG CXXFLAGS="-O3 -static-libgcc -fno-strict-overflow -fstack-protector-all -fPIE"
ARG LDFLAGS="-Wl,-z,relro,-z,now"

# master is broken https://github.com/sekrit-twc/zimg/issues/181
# No rule to make target 'graphengine/graphengine/cpuinfo.cpp', needed by 'graphengine/graphengine/libzimg_internal_la-cpuinfo.lo'.  Stop.
RUN wget https://github.com/sekrit-twc/zimg/archive/refs/tags/release-3.0.4.tar.gz && tar -zxvf release-3.0.4.tar.gz && cd zimg-release-3.0.4 && \
  ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

ENV PATH /usr/local/bin:$PATH
RUN wget https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R62.tar.gz && \
  tar -zxvf R62.tar.gz && cd vapoursynth-R62 && ./autogen.sh && \
  PKG_CONFIG_PATH="/usr/lib/pkgconfig:/usr/local/lib/pkgconfig" ./configure --enable-static --disable-shared && \
  make && make install && cd .. && ldconfig

RUN git clone https://github.com/gypified/libmp3lame && cd libmp3lame && ./configure --enable-static --enable-nasm --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/mstorsjo/fdk-aac/ && \
  cd fdk-aac && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/ogg && cd ogg && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/vorbis && cd vorbis && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/opus && cd opus && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/theora && cd theora && ./autogen.sh && ./configure --disable-examples --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/webmproject/libvpx/ && \
  cd libvpx && ./configure --enable-static --enable-vp9-highbitdepth --disable-shared --disable-unit-tests --disable-examples && \
  make -j$(nproc) install

RUN git clone https://code.videolan.org/videolan/x264.git && \
  cd x264 && ./configure --enable-pic --enable-static && make -j$(nproc) install

# -w-macro-params-legacy to not log lots of asm warnings
# https://bitbucket.org/multicoreware/x265_git/issues/559/warnings-when-assembling-with-nasm-215
RUN git clone https://bitbucket.org/multicoreware/x265_git/ && cd x265_git/build/linux && \
  cmake -G "Unix Makefiles" -DENABLE_SHARED=OFF -D HIGH_BIT_DEPTH:BOOL=ON -DENABLE_AGGRESSIVE_CHECKS=ON ../../source -DCMAKE_ASM_NASM_FLAGS=-w-macro-params-legacy && \
  make -j$(nproc) install

RUN git clone https://github.com/webmproject/libwebp/ && \
  cd libwebp && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/speex/ && \
  cd speex && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone --depth 1 https://aomedia.googlesource.com/aom && \
  cd aom && \
  mkdir build_tmp && cd build_tmp && cmake -DBUILD_SHARED_LIBS=0 -DENABLE_TESTS=0 -DENABLE_NASM=on -DCMAKE_INSTALL_LIBDIR=lib .. && make -j$(nproc) install

RUN git clone https://github.com/georgmartius/vid.stab/ && \
  cd vid.stab && cmake -DBUILD_SHARED_LIBS=OFF . && make -j$(nproc) install

RUN git clone https://github.com/ultravideo/kvazaar/ && \
  cd kvazaar && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/libass/libass/ && \
  cd libass && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) && make install

RUN git clone https://github.com/uclouvain/openjpeg/ && \
  cd openjpeg && cmake -G "Unix Makefiles" -DBUILD_SHARED_LIBS=OFF && make -j$(nproc) install

RUN git clone https://code.videolan.org/videolan/dav1d/ && \
  cd dav1d && meson build --buildtype release -Ddefault_library=static && ninja -C build install

# add extra CFLAGS that are not enabled by -O3
# http://websvn.xvid.org/cvs/viewvc.cgi/trunk/xvidcore/build/generic/configure.in?revision=2146&view=markup
ARG XVID_VERSION=1.3.7
ARG XVID_URL="https://downloads.xvid.com/downloads/xvidcore-$XVID_VERSION.tar.gz"
ARG XVID_SHA256=abbdcbd39555691dd1c9b4d08f0a031376a3b211652c0d8b3b8aa9be1303ce2d
RUN wget -O libxvid.tar.gz "$XVID_URL" && \
  echo "$XVID_SHA256  libxvid.tar.gz" | sha256sum --status -c - && \
  tar xf libxvid.tar.gz && \
  cd xvidcore/build/generic && \
  CFLAGS="$CLFAGS -fstrength-reduce -ffast-math" \
    ./configure && make -j$(nproc) && make install

# currently lots of errors, not stable to build
#0 146.5 Error: CliError { error: Some(failed to verify the checksum of `const_fn_assert v0.1.2`), exit_code: 101 }
#0 73.29 Error: CliError { error: Some(failed to verify the checksum of `const_fn_assert v0.1.2+deprecated`), exit_code: 101 }
# https://crates.io/crates/const_fn_assert released v0.1.2+deprecated and probably broke everything

RUN rm -rf rav1e && \
    git clone https://github.com/xiph/rav1e/ && \
    cd rav1e && \
    rm -f Cargo.lock && \
    sed -i 's/const_fn_assert = "0.1.2"/const_fn_assert = "0.1.2+deprecated"/' Cargo.toml && \
    cargo install cargo-c && \
    cargo cinstall --release --library-type=staticlib --crt-static && \
    sed -i 's/-lgcc_s//' /usr/local/lib/pkgconfig/rav1e.pc

RUN git clone https://github.com/Haivision/srt/ && \
  cd srt && ./configure --enable-shared=0 --cmake-install-libdir=lib --cmake-install-includedir=include --cmake-install-bindir=bin && \
  make -j$(nproc) && make install

RUN git clone https://gitlab.com/AOMediaCodec/SVT-AV1/ && \
  cd SVT-AV1 && \
  sed -i 's/picture_copy(/svt_av1_picture_copy(/g' \
    Source/Lib/Common/Codec/EbPictureOperators.c \
    Source/Lib/Common/Codec/EbPictureOperators.h \
    Source/Lib/Encoder/Codec/EbFullLoop.c \
    Source/Lib/Encoder/Codec/EbProductCodingLoop.c && \
  cd Build && \
  cmake .. -G"Unix Makefiles" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && \
  make -j$(nproc) install

RUN git clone https://github.com/pkuvcl/davs2/ && \
  cd davs2/build/linux && ./configure --disable-asm --enable-pic && \
  make -j$(nproc) install

RUN git clone https://github.com/pkuvcl/xavs2/ && \
  cd xavs2/build/linux && ./configure --disable-asm --enable-pic && \
  make -j$(nproc) install

RUN git clone https://github.com/Netflix/vmaf/ && \
  cd vmaf/libvmaf && meson build --buildtype release -Ddefault_library=static && ninja -vC build install

RUN git clone https://github.com/cisco/openh264 && \
  cd openh264 && meson build --buildtype release -Ddefault_library=static && ninja -C build install

RUN git clone https://github.com/FFmpeg/nv-codec-headers && cd nv-codec-headers && make -j$(nproc) && make install

# https://github.com/shadowsocks/shadowsocks-libev/issues/623
RUN mkdir -p "/home/makepkg/ssl"
RUN git clone git://git.openssl.org/openssl.git && cd openssl && LIBS="-ldl -lz" LDFLAGS="-Wl,-static -static -static-libgcc -s" \
  ./config no-shared -static --prefix="/home/makepkg/ssl" --openssldir="/home/makepkg/ssl" && \
  sed -i 's/^LDFLAGS = /LDFLAGS = -all-static -s/g' Makefile && make -j$(nproc) && make install_sw && make install

# https://stackoverflow.com/questions/18185618/how-to-use-static-linking-with-openssl-in-c-c
RUN git clone https://github.com/FFmpeg/FFmpeg && cd FFmpeg && \
  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/:/home/makepkg/ssl/lib64/pkgconfig/ ./configure \
    --pkg-config-flags=--static \
    --extra-cflags="-fopenmp -lcrypto -lz -ldl -static-libgcc" \
    --extra-ldflags="-fopenmp -lcrypto -lz -ldl -static-libgcc" \
    --extra-libs="-lstdc++ -lcrypto -lz -ldl -static-libgcc" \
    --toolchain=hardened \
    --disable-debug \
    --disable-shared \
    --disable-ffplay \
    --enable-static \
    --enable-gpl \
    --enable-gray \
    --enable-nonfree \
    --enable-openssl \
    --enable-iconv \
    --enable-libxml2 \
    --enable-libmp3lame \
    --enable-libfdk-aac \
    --enable-libvorbis \
    --enable-libopus \
    --enable-libtheora \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libwebp \
    --enable-libspeex \
    --enable-libaom \
    --enable-libvidstab \
    --enable-libkvazaar \
    --enable-libfreetype \
    --enable-fontconfig \
    --enable-libfribidi \
    --enable-libass \
    --enable-libsoxr \
    --enable-libopenjpeg \
    --enable-libdav1d \
    --enable-librav1e \
    --enable-libsrt \
    --enable-libsvtav1 \
    --enable-libdavs2 \
    --enable-libxavs2 \
    --enable-libvmaf \
    --enable-cuda-nvcc \
    --extra-cflags=-I/opt/cuda/include --extra-ldflags=-L/opt/cuda/lib64 \
    --enable-vapoursynth \
    #--enable-hardcoded-tables \
    --enable-libopenh264 \
    --enable-optimizations \
    --enable-cuda-llvm \
    --enable-nvdec \
    --enable-nvenc \
    --enable-cuvid \
    --enable-cuda \
    --enable-pthreads \
    --enable-runtime-cpudetect \
    --enable-lto && \
    make -j$(nproc)
  
############################
# MMCV
############################
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as mmcv-ubuntu

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt install wget git python3 python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 -y && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y
RUN pip3 install --upgrade pip
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
# downgrading since broken api
RUN git clone https://github.com/open-mmlab/mmcv --recursive && cd mmcv && git switch 1.x && MMCV_WITH_OPS=1 python setup.py build_ext && \
  MMCV_WITH_OPS=1 python setup.py bdist_wheel

############################
# cupy
############################

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 as cupy-ubuntu

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt install wget git python3 python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 -y && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y
RUN pip3 install --upgrade pip
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
RUN git clone https://github.com/cupy/cupy --recursive && cd cupy && git submodule update --init && pip install . && python setup.py bdist_wheel

############################
# VSGAN
############################

# https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/11.4.2/ubuntu2204/base/Dockerfile
FROM ubuntu:22.04 as base
ARG DEBIAN_FRONTEND=noninteractive

COPY --from=vulkan-khronos /usr/local/bin /usr/local/bin
COPY --from=vulkan-khronos /usr/local/lib /usr/local/lib
COPY --from=vulkan-khronos /usr/local/include/vulkan /usr/local/include/vulkan
COPY --from=vulkan-khronos /usr/local/share/vulkan /usr/local/share/vulkan

COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
FROM base as base-amd64
ARG DEBIAN_FRONTEND=noninteractive
ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.4"
FROM base-${TARGETARCH}
ARG TARGETARCH
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN apt-get update && apt-get install -y --no-install-recommends \
  gnupg2 curl ca-certificates && \
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/3bf863cc.pub | apt-key add - && \
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH} /" >/etc/apt/sources.list.d/cuda.list && \
  apt-get purge --autoremove -y curl && \
  rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
  cuda-12-1 \
  cuda-cudart-12-1 \
  cuda-compat-12-1 && \
  rm -rf /var/lib/apt/lists/*
RUN echo "/usr/local/nvidia/lib" >>/etc/ld.so.conf.d/nvidia.conf && \
  echo "/usr/local/nvidia/lib64" >>/etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_DRIVER_CAPABILITIES all
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libx11-xcb-dev \
  libxkbcommon-dev \
  libwayland-dev \
  libxrandr-dev \
  libegl1-mesa-dev && \
  rm -rf /var/lib/apt/lists/*
# may not be required
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64:/usr/local/cuda-12.0/lib
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_DRIVER_CAPABILITIES all

WORKDIR workspace
# wget
RUN apt-get -y update && apt install wget fftw3-dev python3 python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 -y && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y
RUN pip3 install --upgrade pip

# TensorRT
RUN apt-get update -y && apt-get install libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev libnvonnxparsers-dev \
  libnvparsers-dev libnvinfer-plugin-dev python3-libnvinfer tensorrt python3-libnvinfer-dev -y && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y
# RUN pip install nvidia-pyindex && pip install tensorrt nvidia-tensorrt

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.0-rc1/cmake-3.23.0-rc1-linux-x86_64.sh && \
  chmod +x cmake-3.23.0-rc1-linux-x86_64.sh && sh cmake-3.23.0-rc1-linux-x86_64.sh --skip-license && \
  cp /workspace/bin/cmake /usr/bin/cmake && cp /workspace/bin/cmake /usr/lib/x86_64-linux-gnu/cmake && \
  cp /workspace/bin/cmake /usr/local/bin/cmake && cp -r /workspace/share/cmake-3.23 /usr/local/share/

# installing vapoursynth and torch
RUN apt update -y && \
  #apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa -y && \
  apt install pkg-config wget python3-pip git p7zip-full x264 autoconf libtool yasm ffmsindex libffms2-5 libffms2-dev -y && \
  wget https://github.com/sekrit-twc/zimg/archive/refs/tags/release-3.0.4.zip && 7z x release-3.0.4.zip && \
  cd zimg-release-3.0.4 && ./autogen.sh && ./configure && make -j4 && make install && cd .. && rm -rf zimg-release-3.0.4 release-3.0.4.zip && \
  pip install --upgrade pip && pip install Cython && git clone https://github.com/Setsugennoao/vapoursynth && cd vapoursynth && git switch preset-typo && ./autogen.sh && \
  ./configure && make && make install && cd .. && ldconfig && \
  ln -s /usr/local/lib/python3.10/site-packages/vapoursynth.so /usr/lib/python3.10/lib-dynload/vapoursynth.so && \
  apt install sudo -y && \
  sudo -H MAKEFLAGS="-j$(nproc)" pip install wget cmake scipy mmedit vapoursynth meson ninja numba numpy scenedetect \
    opencv-python opencv-contrib-python pytorch-msssim thop einops nvidia-pyindex tensorrt==8.6.1 kornia mpgg vsutil \
    onnx onnxruntime-gpu && pip install pycuda && pip3 install polygraphy && \
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118 --force-reinstall -U && \
  git clone https://github.com/pytorch/TensorRT --recursive && cd TensorRT/py && python3 setup.py install --fx-only && cd .. && cd .. && rm -rf TensorRT && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# color transfer
RUN apt install sudo -y && sudo -H pip install docutils pygments && git clone https://github.com/hahnec/color-matcher && cd color-matcher && sudo -H pip install . && \
  cd /workspace && rm -rf color-matcher

# installing onnx tensorrt with a workaround, error with import otherwise
# https://github.com/onnx/onnx-tensorrt/issues/643
# also disables pip cache purge
RUN git clone https://github.com/onnx/onnx-tensorrt.git && \
  cd onnx-tensorrt && \
  cp -r onnx_tensorrt /usr/local/lib/python3.10/dist-packages && \
  cd .. && rm -rf onnx-tensorrt

# imagemagick for imread
RUN apt-get install checkinstall libwebp-dev libopenjp2-7-dev librsvg2-dev libde265-dev -y && git clone https://github.com/ImageMagick/ImageMagick && cd ImageMagick && \
  ./configure --enable-shared --with-modules --with-gslib && make -j$(nproc) && \
  make install && ldconfig /usr/local/lib && cd /workspace && rm -rf ImageMagick && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

# installing tensorflow because of FILM
RUN sudo -H pip install tensorflow tensorflow_addons gin-config

# vs plugings from others
# https://github.com/HolyWu/vs-swinir
# https://github.com/HolyWu/vs-basicvsrpp
RUN sudo -H pip install vsswinir vsbasicvsrpp
# modified versions which take holywu code as a base
RUN git clone https://github.com/styler00dollar/vs-gmfss_union && cd vs-gmfss_union && pip install . && cd /workspace && rm -rf vs-gmfss_union
RUN git clone https://github.com/styler00dollar/vs-gmfss_fortuna && cd vs-gmfss_fortuna && pip install . && cd /workspace && rm -rf vs-gmfss_fortuna
RUN git clone https://github.com/styler00dollar/vs-dpir && cd vs-dpir && pip install . && cd .. && rm -rf vs-dpir

# vs-mlrt
# upgrading g++
RUN apt install build-essential manpages-dev software-properties-common -y && add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
  apt update -y && apt install gcc-11 g++-11 -y && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
  # compiling
  git clone https://github.com/AmusementClub/vs-mlrt /workspace/vs-mlrt && cd /workspace/vs-mlrt/vstrt && mkdir build && \
  cd build && cmake .. -DVAPOURSYNTH_INCLUDE_DIRECTORY=/workspace/vapoursynth/include -D USE_NVINFER_PLUGIN=ON && make -j$(nproc) && make install && \
  cd /workspace && rm -rf /workspace/vs-mlrt

# descale
RUN git clone https://github.com/Irrational-Encoding-Wizardry/descale && cd descale && meson build && ninja -C build && ninja -C build install && \
  cd .. && rm -rf descale

# mpv
RUN apt install mpv -y && apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --yes pulseaudio-utils && \
  apt-get install -y pulseaudio && apt-get install pulseaudio libpulse-dev osspd -y && \
  apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

########################
# vulkan
RUN apt install vulkan-tools libvulkan1 libvulkan-dev -y && apt-get autoclean -y && apt-get autoremove -y && apt-get clean -y

RUN wget https://sdk.lunarg.com/sdk/download/1.3.239.0/linux/vulkansdk-linux-x86_64-1.3.239.0.tar.gz && tar -zxvf vulkansdk-linux-x86_64-1.3.239.0.tar.gz && \
  rm -rf vulkansdk-linux-x86_64-1.3.239.0.tar.gz
ENV VULKAN_SDK=/workspace/1.3.239.0/x86_64/

# rife ncnn
RUN apt install nasm -y && wget https://github.com/Netflix/vmaf/archive/refs/tags/v2.3.1.tar.gz && \
  # VMAF
  tar -xzf v2.3.1.tar.gz && cd vmaf-2.3.1/libvmaf/ && \
  meson build --buildtype release && ninja -C build && \
  ninja -C build install && cd /workspace && rm -rf v2.3.1.tar.gz vmaf-2.3.1 && \
  git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF && cd VapourSynth-VMAF && meson build && \
  ninja -C build && ninja -C build install && cd /workspace && rm -rf VapourSynth-VMAF && \

  # MISC
  git clone https://github.com/vapoursynth/vs-miscfilters-obsolete && cd vs-miscfilters-obsolete && meson build && \
  ninja -C build && ninja -C build install && cd /workspace && rm -rf vs-miscfilters-obsolete && \

  # RIFE
  git clone https://github.com/styler00dollar/VapourSynth-RIFE-ncnn-Vulkan && cd VapourSynth-RIFE-ncnn-Vulkan && \
  git submodule update --init --recursive --depth 1 && meson build && ninja -C build && ninja -C build install && \
  cd /workspace && rm -rf VapourSynth-RIFE-ncnn-Vulkan

########################
# vs plugins

# Vapoursynth-VFRToCFR
RUN git clone https://github.com/Irrational-Encoding-Wizardry/Vapoursynth-VFRToCFR && cd Vapoursynth-VFRToCFR && \
  mkdir build && cd build && meson --buildtype release .. && ninja && ninja install && cd /workspace && rm -rf Vapoursynth-VFRToCFR

# vapoursynth-mvtools
RUN git clone https://github.com/dubhater/vapoursynth-mvtools && cd vapoursynth-mvtools && ./autogen.sh && ./configure && make -j$(nproc) && make install && \
  cd /workspace && rm -rf vapoursynth-mvtools

# fmtconv
RUN git clone https://github.com/EleonoreMizo/fmtconv && cd fmtconv/build/unix/ && ./autogen.sh && ./configure && make -j$(nproc) && make install && \
  cd /workspace && rm -rf fmtconv

# akarin vs
RUN apt install llvm-12 llvm-12-dev -y && git clone https://github.com/AkarinVS/vapoursynth-plugin && \
  cd vapoursynth-plugin && meson build && ninja -C build && \
  ninja -C build install && cd /workspace && rm -rf vapoursynth-plugin

# lsmash
# compiling ffmpeg because apt packages are too old (ffmpeg4.4 because 5 fails to compile)
# but branch ffmpeg-4.5 compiles with ffmpeg5 for whatever reason
# using shared to avoid -fPIC https://ffmpeg.org/pipermail/libav-user/2014-December/007720.html
# RUN git clone https://github.com/FFmpeg/FFmpeg && cd FFmpeg && git switch release/4.4 && git checkout de1132a89113b131831d8edde75214372c983f32
RUN git clone https://github.com/FFmpeg/FFmpeg && cd FFmpeg && \
  CFLAGS=-fPIC ./configure --enable-shared --disable-static --enable-pic && make -j$(nproc) && make install && ldconfig && \
  cd /workspace && rm -rf FFmpeg && git clone https://github.com/l-smash/l-smash && cd l-smash && CFLAGS=-fPIC ./configure --disable-static --enable-shared && \
  make -j$(nproc) && make install && cd /workspace && rm -rf l-smash && git clone https://github.com/AkarinVS/L-SMASH-Works && cd L-SMASH-Works && \
  git switch ffmpeg-4.5 && cd VapourSynth/ && meson build && ninja -C build && ninja -C build install && cd /workspace && rm -rf L-SMASH-Works && ldconfig

# julek
RUN apt install clang libstdc++-12-dev -y
RUN git clone https://github.com/dnjulek/vapoursynth-julek-plugin --recurse-submodules -j8 && cd vapoursynth-julek-plugin/thirdparty && \
  mkdir libjxl_build && cd libjxl_build && cmake -C ../libjxl_cache.cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -G Ninja ../libjxl && \
  cmake --build . && cmake --install . && cd ../.. && cmake -DCMAKE_CXX_COMPILER=clang++ -B build -DCMAKE_BUILD_TYPE=Release -G Ninja && \
  cmake --build build && cmake --install build && cd /workspace && rm -rf vapoursynth-julek-plugin

# warpsharp
RUN git clone https://github.com/dubhater/vapoursynth-awarpsharp2 && cd vapoursynth-awarpsharp2 && mkdir build && \
  cd build && meson ../ && ninja && ninja install && cd /workspace && rm -rf vapoursynth-awarpsharp2

# CAS
RUN git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS && cd VapourSynth-CAS && meson build && \
  ninja -C build && ninja -C build install && cd .. && rm -rf VapourSynth-CAS

# deleting files
RUN rm -rf 1.3.239.0 cmake-3.23.0-rc1-linux-x86_64.sh zimg vapoursynth

# move trtexec so it can be globally accessed
RUN mv /usr/src/tensorrt/bin/trtexec /usr/bin

# RealBasicVSR_x4 will download this if you dont download it prior
#RUN wget "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" -P /root/.cache/torch/hub/checkpoints/

########################

# installing own versions of mmcv, cupy and opencv
# docker apperantly does not provide a simple way to share env variables between stages with different containers https://github.com/moby/moby/issues/37345
# installing whl with *
COPY --from=mmcv-ubuntu /mmcv/dist/ /workspace
COPY --from=cupy-ubuntu /cupy/dist/ /workspace
RUN pip uninstall -y mmcv* cupy* $(pip freeze | grep '^opencv' | cut -d = -f 1) && pip install *.whl --force-reinstall && rm -rf dist *.whl

# install custom opencv (pip install . fails currently)
RUN pip install scikit-build && \
    git clone --recursive https://github.com/opencv/opencv-python.git && \
    cd opencv-python && \
    git submodule update --init --recursive && \
    git submodule update --remote --merge && \
    CMAKE_ARGS="-DOPENCV_EXTRA_MODULES_PATH=/workspace/opencv-python/opencv_contrib/modules \
    -D BUILD_TIFF=ON \
    -D BUILD_opencv_java=OFF \
    -D WITH_CUDA=ON \
    -D WITH_OPENGL=ON \
    -D WITH_OPENCL=ON \
    -D WITH_IPP=ON \
    -D WITH_TBB=ON \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=OFF  \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D OPENCV_FFMPEG_USE_FIND_PACKAGE=ON \
    -D BUILD_SHARED_LIBS=OFF \
    -D CUDA_ARCH_BIN=7.5,8.0,8.6,8.9,7.5+PTX,8.0+PTX,8.6+PTX,8.9+PTX \
    -D CMAKE_BUILD_TYPE=RELEASE" \
    ENABLE_CONTRIB=1 MAKEFLAGS="-j$(nproc)" \
    python setup.py bdist_wheel --verbose && \
    cd dist && \
    pip install *.whl --force-reinstall && \
    cd .. && \
    cd .. && \
    rm -rf opencv-python
    
########################
# av1an
RUN apt install curl libssl-dev mkvtoolnix mkvtoolnix-gui clang-12 nasm libavutil-dev libavformat-dev libavfilter-dev -y && apt-get autoremove -y && apt-get clean
ENV PATH="/root/.cargo/bin:$PATH"

# av1an
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
  . $HOME/.cargo/env && \
  git clone https://github.com/styler00dollar/Av1an && \
  cd Av1an && cargo build --release --features ffmpeg_static && \
  mv /workspace/Av1an/target/release/av1an /usr/bin && \
  cd /workspace && rm -rf Av1an

RUN git clone https://code.videolan.org/videolan/x264.git && \
  cd x264 && ./configure --enable-pic --enable-static --enable-avx512 && make -j$(nproc) install && cd .. && rm -rf x264

# -w-macro-params-legacy to not log lots of asm warnings
# https://bitbucket.org/multicoreware/x265_git/issues/559/warnings-when-assembling-with-nasm-215
RUN git clone https://bitbucket.org/multicoreware/x265_git/ && cd x265_git/build/linux && \
  cmake -G "Unix Makefiles" -DCMAKE_C_FLAGS="-mavx512f" -DCMAKE_CXX_FLAGS="-mavx512f" -DENABLE_SHARED=OFF -DENABLE_AGGRESSIVE_CHECKS=ON \
    ../../source -DCMAKE_ASM_NASM_FLAGS=-w-macro-params-legacy && make -j$(nproc) install && cd /workspace/ && rm -rf x265_git

RUN git clone https://github.com/xiph/rav1e && \
  cd rav1e && \
  cargo build --release && \
  strip ./target/release/rav1e && \
  mv ./target/release/rav1e /usr/local/bin && \
  cd .. && rm -rf ./rav1e

RUN git clone https://gitlab.com/AOMediaCodec/SVT-AV1/ && \
  cd SVT-AV1 && \
  sed -i 's/picture_copy(/svt_av1_picture_copy(/g' \
    Source/Lib/Common/Codec/EbPictureOperators.c \
    Source/Lib/Common/Codec/EbPictureOperators.h \
    Source/Lib/Encoder/Codec/EbFullLoop.c \
    Source/Lib/Encoder/Codec/EbProductCodingLoop.c && \
  cd Build && \
  cmake .. -G"Unix Makefiles" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && \
  make -j$(nproc) install && cd /workspace && rm -rf SVT-AV1

RUN git clone --depth 1 https://aomedia.googlesource.com/aom && \
  cd aom && \
  mkdir build_tmp && cd build_tmp && cmake -DCMAKE_CXX_FLAGS="-O3 -march=native -pipe" -DBUILD_SHARED_LIBS=0 \
  -DENABLE_TESTS=0 -DENABLE_NASM=on -DCMAKE_INSTALL_LIBDIR=lib .. && make -j$(nproc) install && cd /workspace && rm -rf aom

COPY --from=ffmpeg-arch /home/makepkg/FFmpeg/ffmpeg /usr/local/bin/ffmpeg

ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace/tensorrt

# windows hotfix
RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia* /usr/lib/x86_64-linux-gnu/libcuda*
