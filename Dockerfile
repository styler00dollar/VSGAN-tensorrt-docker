############################
# FFMPEG
############################
FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu26.04 AS ffmpeg
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt update -y
RUN apt install autoconf libtool nasm ninja-build yasm wget git pkg-config -y

# -O3 makes sure we compile with optimization. setting CFLAGS/CXXFLAGS seems to override
# default automake cflags.
# -static-libgcc is needed to make gcc not include gcc_s as "as-needed" shared library which
# cmake will include as a implicit library.
# other options to get hardened build (same as ffmpeg hardened)
ARG CFLAGS="-O3 -static-libgcc -fno-strict-overflow -fstack-protector-all -fPIE"
ARG CXXFLAGS="-O3 -static-libgcc -fno-strict-overflow -fstack-protector-all -fPIE"
ARG LDFLAGS="-Wl,-z,relro,-z,now"

RUN git clone https://github.com/gypified/libmp3lame && cd libmp3lame && ./configure --enable-static --enable-nasm --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/mstorsjo/fdk-aac/ && \
  cd fdk-aac && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/ogg && cd ogg && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/vorbis && cd vorbis && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/xiph/opus && cd opus && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN git clone https://github.com/webmproject/libvpx/ && \
  cd libvpx && ./configure --enable-static --enable-vp9-highbitdepth --disable-shared --disable-unit-tests --disable-examples && \
  make -j$(nproc) install

RUN git clone https://code.videolan.org/videolan/x264.git && \
  cd x264 && ./configure --enable-pic --enable-static && make -j$(nproc) install

# cmake
RUN apt-get -y update && apt install wget && wget https://github.com/Kitware/CMake/releases/download/v4.5.0/cmake-4.5.0-linux-x86_64.sh && \
    chmod +x cmake-4.5.0-linux-x86_64.sh && sh cmake-4.5.0-linux-x86_64.sh --skip-license && \
    cp /workspace/bin/cmake /usr/bin/cmake && cp /workspace/bin/cmake /usr/lib/cmake && \
    cp /workspace/bin/cmake /usr/local/bin/cmake && cp -r /workspace/share/cmake-4.5 /usr/local/share/ && \
    rm -rf cmake-4.5.0-linux-x86_64.sh
# -w-macro-params-legacy to not log lots of asm warnings
# https://bitbucket.org/multicoreware/x265_git/issues/559/warnings-when-assembling-with-nasm-215
RUN git clone https://bitbucket.org/multicoreware/x265_git/ && cd x265_git/build/linux && \
  cmake -G "Unix Makefiles" -DENABLE_SHARED=OFF -D HIGH_BIT_DEPTH:BOOL=ON -DENABLE_AGGRESSIVE_CHECKS=ON ../../source -DCMAKE_ASM_NASM_FLAGS=-w-macro-params-legacy && \
  make -j$(nproc) install

RUN git clone https://github.com/webmproject/libwebp/ && \
  cd libwebp && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

RUN apt install libfreetype-dev libfribidi-dev libharfbuzz-dev libfontconfig-dev -y
RUN git clone https://github.com/libass/libass/ && \
  cd libass && ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) && make install

RUN apt install python3-pip -y
RUN pip install meson ninja cython --break-system-packages
RUN git clone https://code.videolan.org/videolan/dav1d/ && \
  cd dav1d && meson build --buildtype release -Ddefault_library=static && ninja -C build install

RUN git clone https://github.com/nekotrix/SVT-AV1-Essential && \
  cd SVT-AV1-Essential && git switch Essential-v4.3.0 && \
  cd Build/linux && \
  ./build.sh --native --static --release --enable-lto --enable-pgo --install

RUN git clone https://github.com/FFmpeg/nv-codec-headers && cd nv-codec-headers && make -j$(nproc) && make install

RUN git clone https://github.com/FFmpeg/FFmpeg

RUN cd FFmpeg && ./configure \
    --extra-cflags="-march=native" \
    --extra-cxxflags="-march=native" \
    --extra-ldflags="-static-libgcc" \
    --extra-libs="-static-libgcc" \
    --pkg-config-flags="--static" \
    --toolchain=hardened \
    --disable-debug \
    --disable-shared \
    --disable-ffplay \
    --enable-static \
    --enable-gpl \
    --enable-gray \
    --enable-nonfree \
    --disable-openssl \
    --enable-iconv \
    --disable-libxml2 \
    --enable-libmp3lame \
    --enable-libfdk-aac \
    --enable-libvorbis \
    --enable-libopus \
    --disable-libtheora \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libwebp \
    --disable-libspeex \
    --disable-libaom \
    --disable-libvidstab \
    --enable-libfreetype \
    --enable-fontconfig \
    --enable-libfribidi \
    --enable-libass \
    --disable-libsoxr \
    --enable-libdav1d \
    --disable-librav1e \
    --enable-libsvtav1 \
    --disable-libdavs2 \
    --disable-libxeve \
    --enable-cuda-nvcc \
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
# trtexec
############################

FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu26.04 AS trtexec
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt update -y && apt install wget zstd -y
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.4.0/tars/TensorRT-Enterprise-11.4.0.106-Linux-x86_64-cuda-13.4-Release-external.tar.zst
RUN mkdir TensorRT && tar --use-compress-program=unzstd -xvf TensorRT-Enterprise-11.4.0.106-Linux-x86_64-cuda-13.4-Release-external.tar.zst -C TensorRT

############################

FROM nvidia/cuda:13.4.3-runtime-ubuntu26.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install python3.14 python3.14-dev python3-pip python-is-python3 git clang -y
RUN pip install git+https://github.com/vapoursynth/vapoursynth.git vapoursynth-mlrt-trt vapoursynth-bestsource vapoursynth-ffms2 \
  onnxruntime-gpu vapoursynth-akarin vapoursynth-awarp vapoursynth-cas vsrife vsrealesrgan vs-colorfix vs-temporalfix vapoursynth-descale tensorrt --break-system-packages
RUN python -m vsrife && python -m vsrealesrgan

RUN apt update -y && apt install fontconfig libharfbuzz0b libfribidi0 -y

# ffmpeg
COPY --from=ffmpeg /workspace/FFmpeg/ffmpeg /usr/local/bin/ffmpeg

# trtexec
COPY --from=trtexec /workspace/TensorRT/TensorRT-11.4.0.106/bin/trtexec /usr/local/bin/trtexec
RUN ln -s /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer.so.11 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_plugin.so.11 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvonnxparser.so.11 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_ptx.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm75.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm80.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm86.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm89.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm90.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm100.so.11.4.0 \
  /usr/local/lib/python3.14/dist-packages/tensorrt_libs/libnvinfer_builder_resource_sm120.so.11.4.0 \
  /usr/local/lib

RUN pip install vapoursynth-edgemasks torch-tensorrt cuda-python --break-system-packages

ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace/tensorrt
