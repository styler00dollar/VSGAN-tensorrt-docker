############################
# FFMPEG
############################
FROM archlinux AS ffmpeg-arch
RUN --mount=type=cache,sharing=locked,target=/var/cache/pacman \
  pacman -Syu --noconfirm --needed base base-devel cuda git
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ARG user=makepkg
RUN useradd --system --create-home $user && \
  echo "$user ALL=(ALL:ALL) NOPASSWD:ALL" >/etc/sudoers.d/$user
USER $user
WORKDIR /home/$user
RUN git clone https://aur.archlinux.org/yay-bin.git && \
  cd yay-bin && \
  makepkg -si --noconfirm

RUN yay -Syu rust tcl nasm cmake jq libtool wget fribidi fontconfig libsoxr meson pod2man nvidia-utils base-devel --noconfirm --ask 4
RUN yay -S python-pip python312 --noconfirm --ask 4

USER root

RUN mkdir -p "/home/makepkg/python312"
RUN wget https://github.com/python/cpython/archive/refs/tags/v3.12.7.tar.gz && tar xf v3.12.7.tar.gz && cd cpython-3.12.7 && \
  mkdir debug && cd debug && ../configure --enable-optimizations --disable-shared --prefix="/home/makepkg/python312" && make -j$(nproc) && make install && \
  /home/makepkg/python312/bin/python3.12 -m ensurepip --upgrade
RUN cp /home/makepkg/python312/bin/python3.12 /usr/bin/python
ENV PYTHONPATH /home/makepkg/python312/bin/
ENV PATH "/home/makepkg/python312/bin/:$PATH"

RUN pip3 install "cython<3" meson

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

# todo: use https://bitbucket.org/the-sekrit-twc/zimg/src/master/
# master is broken https://github.com/sekrit-twc/zimg/issues/181
# No rule to make target 'graphengine/graphengine/cpuinfo.cpp', needed by 'graphengine/graphengine/libzimg_internal_la-cpuinfo.lo'.  Stop.
RUN wget https://github.com/sekrit-twc/zimg/archive/refs/tags/release-3.0.5.tar.gz && tar -zxvf release-3.0.5.tar.gz && cd zimg-release-3.0.5 && \
  ./autogen.sh && ./configure --enable-static --disable-shared && make -j$(nproc) install

ENV PATH /usr/local/bin:$PATH
RUN wget https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R70.tar.gz && \
  tar -zxvf R70.tar.gz && cd vapoursynth-R70 && ./autogen.sh && \
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

RUN rm -rf rav1e && \
    git clone https://github.com/xiph/rav1e/ && \
    cd rav1e && \
    cargo install cargo-c && \
    cargo cinstall --release --library-type=staticlib --crt-static && \
    sed -i 's/-lgcc_s//' /usr/local/lib/pkgconfig/rav1e.pc

RUN git clone https://github.com/Haivision/srt/ && \
  cd srt && ./configure --enable-shared=0 --cmake-install-libdir=lib --cmake-install-includedir=include --cmake-install-bindir=bin && \
  make -j$(nproc) && make install

RUN git clone https://github.com/gianni-rosato/svt-av1-psy && \
  cd svt-av1-psy/Build && \
  cmake .. -G"Unix Makefiles" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && \
  make -j$(nproc) install

RUN git clone https://github.com/pkuvcl/davs2/ && \
  cd davs2/build/linux && ./configure --disable-asm --enable-pic && \
  make -j$(nproc) install

RUN git clone https://github.com/Netflix/vmaf/ && \
  cd vmaf/libvmaf && meson build --buildtype release -Ddefault_library=static && ninja -vC build install

RUN git clone https://github.com/cisco/openh264 && \
  cd openh264 && meson build --buildtype release -Ddefault_library=static && ninja -C build install

RUN git clone https://github.com/FFmpeg/nv-codec-headers && cd nv-codec-headers && make -j$(nproc) && make install

RUN git clone https://github.com/mpeg5/xeve && cd xeve && mkdir build && cd build && cmake .. && make -j$(nproc) && make install
RUN rm -rf /usr/local/lib/libxeve.so*

# https://github.com/shadowsocks/shadowsocks-libev/issues/623
RUN mkdir -p "/home/makepkg/ssl"
RUN git clone https://github.com/openssl/openssl && cd openssl && LIBS="-ldl -lz" LDFLAGS="-Wl,-static -static -static-libgcc -s" \
  ./config no-shared -static --prefix="/home/makepkg/ssl" --openssldir="/home/makepkg/ssl" && \
  sed -i 's/^LDFLAGS = /LDFLAGS = -all-static -s/g' Makefile && make -j$(nproc) && make install_sw && make install

# todo: can't figure out "ERROR: failed checking for nvcc", may be gcc version mismatch
# https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version
# https://github.com/NVIDIA/cuda-samples/issues/46#issuecomment-2030697518
# openssl static
# https://stackoverflow.com/questions/18185618/how-to-use-static-linking-with-openssl-in-c-c

RUN git clone https://github.com/FFmpeg/FFmpeg
RUN cd FFmpeg && \
CFLAGS="${CFLAGS} -Wno-incompatible-pointer-types -Wno-implicit-function-declaration" PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/:/home/makepkg/ssl/lib64/pkgconfig/ ./configure \
    --extra-cflags="-march=native -fopenmp -lcrypto -lz -ldl -static-libgcc -I/opt/cuda/include" \
    --extra-cxxflags="-march=native -fopenmp -lcrypto -lz -ldl -static-libgcc" \
    --extra-ldflags="-L/usr/local/lib/xeve -fopenmp -lcrypto -lz -ldl -static-libgcc -L/opt/cuda/lib64" \
    --extra-libs="-lstdc++ -lcrypto -lz -ldl -static-libgcc" \
    --pkg-config-flags=--static \
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
    --enable-libvmaf \
    --enable-libxeve \
    #--enable-cuda-nvcc \ # ERROR: failed checking for nvcc
    --enable-vapoursynth \
    #--enable-hardcoded-tables \
    --enable-libopenh264 \
    --enable-optimizations \
    #--enable-cuda-llvm \ # ERROR: cuda_llvm requested but not found
    --enable-nvdec \
    --enable-nvenc \
    --enable-cuvid \
    --enable-cuda \
    --enable-pthreads \
    --enable-runtime-cpudetect \
    --enable-lto && \
    #--enable-vulkan && \ # currently can't get it working
    make -j$(nproc)
  
############################
# torch
# compiling own torch since the official whl is bloated
# could be smaller in terms of dependencies and whl size, but for now, -500mb smaller docker size
############################
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS torch-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y \
  curl \
  make \
  gcc \
  wget \
  libssl-dev \
  libffi-dev \
  libopenblas-dev \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  python3-pip \
  git && \
  apt-get autoclean -y && \
  apt-get autoremove -y && \
  apt-get clean -y

RUN python3.12 -m pip install numpy pyyaml --break-system-packages
RUN git clone -b release/2.5 --recursive https://github.com/pytorch/pytorch

WORKDIR /cmake

# cmake 3.28 (CMake 3.18.0 or higher is required)
RUN apt-get -y update && apt install wget && wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh && \
    chmod +x cmake-3.30.2-linux-x86_64.sh  && sh cmake-3.30.2-linux-x86_64.sh  --skip-license && \
    cp /cmake/bin/cmake /usr/bin/cmake && cp /cmake/bin/cmake /usr/lib/cmake && \
    cp /cmake/bin/cmake /usr/local/bin/cmake && cp /cmake/bin/ctest /usr/local/bin/ctest && cp -r /cmake/share/cmake-3.30 /usr/local/share/ && \
    rm -rf cmake-3.30.2-linux-x86_64.sh 

WORKDIR /

#RUN cd pytorch && pip3 install -r requirements.txt --break-system-packages && \
#  MAX_JOBS=6 USE_CUDA=1 USE_CUDNN=1 TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.9" USE_NCCL=OFF python3.12 setup.py build && \
#  MAX_JOBS=6 USE_CUDA=1 USE_CUDNN=1 TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.9" python3.12 setup.py bdist_wheel

############################
# cupy
############################
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS cupy-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y \
  curl \
  make \
  gcc \
  wget \
  libssl-dev \
  libffi-dev \
  libopenblas-dev \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  python3-pip \
  git && \
  apt-get autoclean -y && \
  apt-get autoremove -y && \
  apt-get clean -y

RUN python3.12 -m pip install git+https://github.com/numpy/numpy torch torchvision torchaudio --break-system-packages
RUN git clone https://github.com/cupy/cupy --recursive && cd cupy && git submodule update --init && python3.12 -m pip install . --break-system-packages && \
  MAKEFLAGS="-j$(nproc)" python3.12 setup.py bdist_wheel

############################
# mlrt ort
############################
# https://github.com/AmusementClub/vs-mlrt/blob/master/.github/workflows/linux-ort.yml
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS vsort-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y \
  curl \
  make \
  gcc \
  wget \
  libssl-dev \
  libffi-dev \
  libopenblas-dev \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  python3-pip \
  git && \
  apt-get autoclean -y && \
  apt-get autoremove -y && \
  apt-get clean -y

# cmake
WORKDIR /cmake
RUN apt-get -y update && apt install wget && wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh && \
  chmod +x cmake-3.30.2-linux-x86_64.sh  && sh cmake-3.30.2-linux-x86_64.sh  --skip-license && \
  cp /cmake/bin/cmake /usr/bin/cmake && cp /cmake/bin/cmake /usr/lib/cmake && \
  cp /cmake/bin/cmake /usr/local/bin/cmake && cp /cmake/bin/ctest /usr/local/bin/ctest && cp -r /cmake/share/cmake-3.30 /usr/local/share/ && \
  rm -rf cmake-3.30.2-linux-x86_64.sh 

WORKDIR /workdir
# own fork with additional tensorrt backend
RUN git clone https://github.com/styler00dollar/vs-mlrt
WORKDIR /workdir/vs-mlrt

# protobuf
RUN git clone https://github.com/protocolbuffers/protobuf && cd protobuf && git submodule update --init --recursive
RUN mkdir -p protobuf/build_rel
RUN cd protobuf/build_rel && cmake .. -DCMAKE_CXX_STANDARD=17 \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
  -D protobuf_BUILD_SHARED_LIBS=OFF -D protobuf_BUILD_TESTS=OFF
RUN cd protobuf/build_rel && cmake --build . --verbose -j$(nproc) --target install

# onnx
RUN git clone https://github.com/onnx/onnx --recursive
RUN mkdir -p onnx/build
RUN cd onnx/build && cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
  -D ONNX_USE_LITE_PROTO=ON -D ONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
  -D ONNX_GEN_PB_TYPE_STUBS=OFF -D ONNX_ML=0
RUN cd onnx/build && cmake --build . --verbose -j$(nproc) --target install

# vapoursynth
RUN apt install unzip -y
RUN wget -q -O vs.zip https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R57.zip && \
  unzip -q vs.zip && \
  mv vapoursynth*/ vapoursynth

# onnxruntime
RUN wget -O ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-cuda12-1.18.0.tgz && \
  tar -xf ort.tgz && \
  mv onnxruntime-* onnxruntime -v

RUN mkdir -p /workdir/vs-mlrt/vsort/build && cd /workdir/vs-mlrt/vsort/build && cmake .. \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS="-Wall -ffast-math -march=x86-64-v3" \
  -D VAPOURSYNTH_INCLUDE_DIRECTORY="/workdir/vs-mlrt/vapoursynth/include" \
  -D ONNX_RUNTIME_API_DIRECTORY=/workdir/vs-mlrt/onnxruntime/include \
  -D ONNX_RUNTIME_LIB_DIRECTORY=/workdir/vs-mlrt/onnxruntime/lib \
  -D ENABLE_CUDA=1 \
  -D ENABLE_TENSORRT=1 \
  -D CUDAToolkit_ROOT=/usr/local/cuda \
  -D CMAKE_CXX_STANDARD=20

RUN cd /workdir/vs-mlrt/vsort/build && cmake --build . --verbose -j$(nproc) --target install

############################
# bestsource / lsmash / ffms2
# todo: check if CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS="-Wl,-Bsymbolic" --extra-ldflags="-static" is required
############################
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS bestsource-lsmash-ffms2-vs

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR workspace

RUN apt update -y
RUN apt install autoconf libtool nasm ninja-build yasm python3.12 python3.12-venv python3.12-dev python3-pip wget git pkg-config python-is-python3 -y
RUN apt --fix-broken install
RUN pip install meson ninja cython --break-system-packages

# install g++13
RUN apt install build-essential manpages-dev software-properties-common -y
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y
RUN apt update -y && apt install gcc-13 g++-13 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13

# zimg
# setting pkg version manually since otherwise 'Version' field value '-1': version number is empty
RUN apt-get install checkinstall -y
RUN git clone https://github.com/sekrit-twc/zimg --recursive && cd zimg && \
  ./autogen.sh && CFLAGS=-fPIC CXXFLAGS=-fPIC ./configure --enable-static --disable-shared && make -j$(nproc) && make install
RUN rm -rf /usr/local/share/doc/zimg/ChangeLog /usr/local/share/doc/zimg/COPYING /usr/local/share/doc/zimg/README.md /usr/local/share/doc/zimg/example/ /usr/local/include/zimg* /usr/local/lib/pkgconfig/zimg.pc
RUN cd zimg && checkinstall -y -pkgversion=0.0 && \
  apt install /workspace/zimg/zimg_0.0-1_amd64.deb -y

# vapoursynth
RUN wget https://github.com/vapoursynth/vapoursynth/archive/refs/tags/R70.tar.gz && \
  tar -zxvf R70.tar.gz && mv vapoursynth-R70 vapoursynth && cd vapoursynth && \
  ./autogen.sh && CFLAGS=-fPIC CXXFLAGS=-fPIC ./configure --enable-static --disable-shared && make -j$(nproc) && make install && ldconfig

# dav1d
RUN git clone https://code.videolan.org/videolan/dav1d/ && \
  cd dav1d && meson build --buildtype release -Ddefault_library=static && ninja -C build install

# Vulkan-Headers
RUN apt install cmake -y
RUN git clone https://github.com/KhronosGroup/Vulkan-Headers.git && cd Vulkan-Headers/ && cmake -S . -DBUILD_SHARED_LIBS=OFF -B build/ && cmake --install build

# nv-codec-headers
RUN git clone https://github.com/FFmpeg/nv-codec-headers && cd nv-codec-headers && make -j$(nproc) && make install

# ffmpeg
RUN apt remove ffmpeg -y
RUN git clone https://git.ffmpeg.org/ffmpeg.git --depth 1 && cd ffmpeg && \
  CFLAGS=-fPIC ./configure --enable-libdav1d --enable-cuda --enable-nonfree --disable-shared --enable-static --enable-gpl --enable-version3 --disable-programs --disable-doc --disable-avdevice --disable-swresample --disable-postproc --disable-avfilter --disable-encoders --disable-muxers --disable-debug --enable-pic --extra-ldflags="-static" --extra-cflags="-march=native" && \
  make -j$(nproc) && make install -j$(nproc)

# jansson
RUN git clone https://github.com/akheron/jansson && cd jansson && autoreconf -fi && CFLAGS=-fPIC ./configure --disable-shared --enable-static && \
  make -j$(nproc) && make install

# bzip2
RUN git clone https://github.com/libarchive/bzip2 && cd bzip2 && \
  mkdir build && cd build && cmake .. -DBUILD_SHARED_LIBS=OFF && make -j$(nproc) && make install

# bestsource (custom bestsource to add back _AbsoluteTime)
RUN apt-get install libxxhash-dev -y && git clone https://github.com/styler00dollar/bestsource --depth 1 --recurse-submodules --shallow-submodules && cd bestsource && \
  CFLAGS=-fPIC meson setup -Denable_plugin=true build && CFLAGS=-fPIC ninja -C build && ninja -C build install

# ffmpeg (HomeOfAviSynthPlusEvolution version with sws)
# official ffmpeg does not compile
# fatal error: libswresample/swresample.h: No such file or directory
RUN apt remove ffmpeg -y
RUN rm -rf FFmpeg

RUN git clone https://github.com/HomeOfAviSynthPlusEvolution/FFmpeg
RUN cd FFmpeg && \
  LDFLAGS="-Wl,-Bsymbolic" CFLAGS=-fPIC ./configure --enable-libdav1d --enable-cuda --enable-nonfree --disable-shared --enable-static --enable-gpl --enable-version3 --disable-programs --disable-doc --disable-avdevice --disable-postproc --disable-avfilter --disable-encoders --disable-muxers --disable-debug --enable-pic --extra-ldflags="-Wl,-Bsymbolic" --extra-cflags="-march=native" && \
  make -j$(nproc) && make install -j$(nproc)

# lsmash
RUN git clone https://github.com/l-smash/l-smash && cd l-smash && CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS="-Wl,-Bsymbolic" ./configure --enable-shared --extra-ldflags="-Wl,-Bsymbolic"  && \
  make -j$(nproc) && make install
RUN git clone https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works && cd L-SMASH-Works && \
   cd VapourSynth/ && CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS="-Wl,-Bsymbolic" meson build && CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS="-Wl,-Bsymbolic" ninja -C build && ninja -C build install 

# ffms2
RUN apt install autoconf -y
RUN git clone https://github.com/FFMS/ffms2 && cd ffms2 && ./autogen.sh && CFLAGS=-fPIC CXXFLAGS=-fPIC LDFLAGS="-Wl,-Bsymbolic" ./configure --enable-shared && make -j$(nproc) && make install

############################
# TensorRT + ORT
############################
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04 AS tensorrt-ubuntu

ARG DEBIAN_FRONTEND=noninteractive

# install python
# https://stackoverflow.com/questions/75159821/installing-python-3-11-1-on-a-docker-container
# https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in
# /usr/local/lib/libpython3.12.a(longobject.o): relocation R_X86_64_PC32 against symbol `_Py_NotImplementedStruct' can not be used when making a shared object; recompile with -fPIC
# todo: test CFLAGS="-fPIC -march=native"
RUN apt update -y && apt install liblzma-dev libbz2-dev ca-certificates openssl libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev \
  libpcap-dev software-properties-common wget zlib1g-dev -y && \
  wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tar.xz && \
  tar -xf Python-3.12.7.tar.xz && cd Python-3.12.7 && \
  CFLAGS=-fPIC ./configure --with-openssl-rpath=auto --enable-optimizations CFLAGS=-fPIC && \
  make -j$(nproc) && make altinstall && make install
# todo: update-alternatives may not be required
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1 && \
  update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1 && \
  cp /usr/local/bin/python3.12 /usr/local/bin/python && \
  cp /usr/local/bin/pip3.12 /usr/local/bin/pip && \
  cp /usr/local/bin/pip3.12 /usr/local/bin/pip3

# required since ModuleNotFoundError: No module named 'pip' with nvidia pip packages, even if cli works
RUN wget "https://bootstrap.pypa.io/get-pip.py" && python get-pip.py --force-reinstall

# TensorRT10
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/common/install_tensorrt.sh
# https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=tensorrt
RUN wget "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz" -O /tmp/TensorRT.tar
RUN tar -xf /tmp/TensorRT.tar -C /usr/local/
RUN mv /usr/local/TensorRT-10.7.0.23 /usr/local/tensorrt
RUN pip3 install /usr/local/tensorrt/python/tensorrt-*-cp312-*.whl
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/

# cudnn
# https://gitlab.archlinux.org/archlinux/packaging/packages/cudnn/-/blob/main/PKGBUILD?ref_type=heads
# todo: not using 9 because of "Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.8", most likely 9 is usable with custom compiled onnxruntime
#RUN wget "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.2.1.18_cuda12-archive.tar.xz" -O /tmp/cudnn.tar
RUN wget "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz" -O /tmp/cudnn.tar
RUN tar -xf /tmp/cudnn.tar -C /usr/local/
RUN mv /usr/local/cudnn-linux-x86_64-8.9.7.29_cuda12-archive /usr/local/cudnn
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn

# ORT
# onnxruntime requires working tensorrt installation and thus can't be easily seperated into a seperate instance
# https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.tensorrt
ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=rel-1.20.0
ARG CMAKE_CUDA_ARCHITECTURES=37;50;52;53;60;61;62;70;72;75;80;89

RUN apt-get update &&\
    apt-get install -y sudo git bash unattended-upgrades
RUN unattended-upgrade

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt install git -y

# cmake 3.30 (CMake 3.26 or higher is required)
RUN apt-get -y update && apt install wget && wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh && \
    chmod +x cmake-3.30.2-linux-x86_64.sh  && sh cmake-3.30.2-linux-x86_64.sh  --skip-license && \
    cp /code/bin/cmake /usr/bin/cmake && cp /code/bin/cmake /usr/lib/cmake && \
    cp /code/bin/cmake /usr/local/bin/cmake && cp /code/bin/ctest /usr/local/bin/ctest && cp -r /code/share/cmake-3.30 /usr/local/share/ && \
    rm -rf cmake-3.30.2-linux-x86_64.sh 

# Prepare onnxruntime repository & build onnxruntime with TensorRT
# --parallel 6 for 6 compile threads, using all threads ooms my ram
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh 
RUN /usr/local/bin/pip3 install psutil numpy wheel setuptools packaging
RUN cd onnxruntime && PYTHONPATH=/usr/bin/python3 /bin/sh build.sh --nvcc_threads 3 --parallel 4 --allow_running_as_root --build_shared_lib --cuda_home /usr/local/cuda \
      --cudnn_home /usr/local/cudnn --use_tensorrt --tensorrt_home /usr/local/tensorrt --config Release --build_wheel --skip_tests --skip_submodule_sync --cmake_extra_defines '"CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'"'

############################
# VSGAN
############################

# https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/11.4.2/ubuntu2204/base/Dockerfile
FROM ubuntu:24.04 AS base
ARG DEBIAN_FRONTEND=noninteractive
ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.4"
COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN apt-get update && apt-get install -y --no-install-recommends \
  gnupg2 curl ca-certificates && \
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/3bf863cc.pub | apt-key add - && \
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH} /" >/etc/apt/sources.list.d/cuda.list && \
  apt-get purge --autoremove -y curl && \
  rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
  cuda-12-5 \
  cuda-cudart-12-5 \
  cuda-compat-12-5 && \
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
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/lib
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_DRIVER_CAPABILITIES all

WORKDIR workspace

# install python
# https://stackoverflow.com/questions/75159821/installing-python-3-11-1-on-a-docker-container
# https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in
# /usr/local/lib/libpython3.12.a(longobject.o): relocation R_X86_64_PC32 against symbol `_Py_NotImplementedStruct' can not be used when making a shared object; recompile with -fPIC
# todo: test CFLAGS="-fPIC -march=native"
RUN apt update -y && apt install liblzma-dev libbz2-dev ca-certificates openssl libssl-dev libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev \
  libpcap-dev software-properties-common wget zlib1g-dev -y && \
  wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tar.xz && \
  tar -xf Python-3.12.7.tar.xz && cd Python-3.12.7 && \
  CFLAGS=-fPIC ./configure --with-openssl-rpath=auto --enable-optimizations CFLAGS=-fPIC && \
  make -j$(nproc) && make altinstall && make install
# todo: update-alternatives may not be required
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1 && \
  update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1 && \
  cp /usr/local/bin/python3.12 /usr/local/bin/python && \
  cp /usr/local/bin/pip3.12 /usr/local/bin/pip && \
  cp /usr/local/bin/pip3.12 /usr/local/bin/pip3
# required since ModuleNotFoundError: No module named 'pip' with nvidia pip packages, even if cli works
RUN wget "https://bootstrap.pypa.io/get-pip.py" && python get-pip.py --force-reinstall

# python shared (for ffmpeg)
RUN rm -rf Python-3.12.7 && tar -xf Python-3.12.7.tar.xz && cd Python-3.12.7 && \
  CFLAGS=-fPIC ./configure --enable-shared --with-ssl --with-openssl-rpath=auto --enable-optimizations CFLAGS=-fPIC && \
  make -j$(nproc)

# cmake
RUN apt-get -y update && apt install wget && wget https://github.com/Kitware/CMake/releases/download/v3.30.2/cmake-3.30.2-linux-x86_64.sh && \
  chmod +x cmake-3.30.2-linux-x86_64.sh && sh cmake-3.30.2-linux-x86_64.sh --skip-license && \
  cp /workspace/bin/cmake /usr/bin/cmake && cp /workspace/bin/cmake /usr/lib/x86_64-linux-gnu/cmake && \
  cp /workspace/bin/cmake /usr/local/bin/cmake && cp -r /workspace/share/cmake-3.30 /usr/local/share/

# zimg
# setting pkg version manually since otherwise 'Version' field value '-1': version number is empty
RUN apt install python-is-python3 pkg-config python3-pip git p7zip-full autoconf libtool yasm ffmsindex libffms2-5 libffms2-dev -y
RUN apt-get install checkinstall -y
RUN git clone https://github.com/sekrit-twc/zimg --recursive && cd zimg && \
  ./autogen.sh && CFLAGS=-fPIC CXXFLAGS=-fPIC ./configure --enable-static --disable-shared && make -j$(nproc) && make install
RUN rm -rf /usr/local/share/doc/zimg/ChangeLog /usr/local/share/doc/zimg/COPYING /usr/local/share/doc/zimg/README.md /usr/local/share/doc/zimg/example/ /usr/local/include/zimg* /usr/local/lib/pkgconfig/zimg.pc
RUN cd zimg && checkinstall -y -pkgversion=0.0 && \
  apt install /workspace/zimg/zimg_0.0-1_amd64.deb -y

# vapoursynth
RUN pip install --upgrade pip && pip install cython setuptools && git clone https://github.com/vapoursynth/vapoursynth && \
  cd vapoursynth && ./autogen.sh && \
  ./configure && make -j$(nproc) && make install && cd .. && ldconfig && \
  cd vapoursynth && python setup.py bdist_wheel

#################################################################
# color transfer
RUN pip install numpy docutils pygments && git clone https://github.com/hahnec/color-matcher && cd color-matcher && python setup.py bdist_wheel

# vs-mlrt
RUN wget "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz" -O /tmp/TensorRT.tar
RUN tar -xf /tmp/TensorRT.tar -C /usr/local/
RUN mv /usr/local/TensorRT-10.7.0.23/targets/x86_64-linux-gnu/lib/* /usr/lib/x86_64-linux-gnu
RUN mv /usr/local/TensorRT-10.7.0.23/include/* /usr/include/x86_64-linux-gnu/
RUN cd /usr/lib/x86_64-linux-gnu \
  && ldconfig
ENV CPLUS_INCLUDE_PATH="/usr/include/x86_64-linux-gnu/"

# upgrading g++
RUN apt install build-essential manpages-dev software-properties-common -y && add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
  apt update -y && apt install gcc-11 g++-11 -y && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
  # compiling
  git clone https://github.com/AmusementClub/vs-mlrt /workspace/vs-mlrt && cd /workspace/vs-mlrt/vstrt && mkdir build && \
  cd build && cmake .. -DVAPOURSYNTH_INCLUDE_DIRECTORY=/workspace/vapoursynth/include -D USE_NVINFER_PLUGIN=ON && make -j$(nproc) && make install 
# descale
RUN pip install meson ninja && git clone https://github.com/Irrational-Encoding-Wizardry/descale && cd descale && meson build && ninja -C build && ninja -C build install 

########################
# vs plugins
# Vapoursynth-VFRToCFR
RUN git clone https://github.com/Irrational-Encoding-Wizardry/Vapoursynth-VFRToCFR && cd Vapoursynth-VFRToCFR && \
  mkdir build && cd build && meson --buildtype release .. && ninja && ninja install

# fmtconv
RUN git clone https://github.com/EleonoreMizo/fmtconv && cd fmtconv/build/unix/ && ./autogen.sh && ./configure && make -j$(nproc) && make install

# VMAF
RUN apt install nasm xxd -y && wget https://github.com/Netflix/vmaf/archive/refs/tags/v3.0.0.tar.gz && \
  tar -xzf v3.0.0.tar.gz && cd vmaf-3.0.0/libvmaf/ && \
  meson build --buildtype release -Denable_cuda=true -Denable_avx512=true && ninja -C build && \
  ninja -C build install && cd /workspace && rm -rf v3.0.0.tar.gz vmaf-3.0.0 && \
  git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF && cd VapourSynth-VMAF && meson build && \
  ninja -C build && ninja -C build install

# MISC
RUN git clone https://github.com/vapoursynth/vs-miscfilters-obsolete && cd vs-miscfilters-obsolete && meson build && \
  ninja -C build && ninja -C build install

# akarin vs
# official akarin does not support new llvm
# llvm-config found: NO found '18.1.3' but need ['>= 10.0', '< 16']
RUN apt install llvm-18 llvm-18-dev libzstd-dev -y && git clone https://github.com/Jaded-Encoding-Thaumaturgy/akarin-vapoursynth-plugin && \
  cd akarin-vapoursynth-plugin && meson build && ninja -C build && \
  ninja -C build install

# warpsharp
RUN git clone https://github.com/dubhater/vapoursynth-awarpsharp2 && cd vapoursynth-awarpsharp2 && mkdir build && \
  cd build && meson ../ && ninja && ninja install

# CAS
RUN git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS && cd VapourSynth-CAS && meson build && \
  ninja -C build && ninja -C build install 

########################
# av1an
RUN apt install curl libssl-dev mkvtoolnix mkvtoolnix-gui clang nasm libavutil-dev libavformat-dev libavfilter-dev -y && apt-get autoremove -y && apt-get clean
ENV PATH="/root/.cargo/bin:$PATH"

# av1an
# todo: use own custom av1an
# todo: removing everything that isn't ffmpeg?
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
  . $HOME/.cargo/env && \
  git clone https://github.com/master-of-zen/Av1an && \
  cd Av1an && cargo build --release --features ffmpeg_static && \
  mv /workspace/Av1an/target/release/av1an /usr/bin 

RUN git clone https://github.com/xiph/rav1e && \
  cd rav1e && \
  cargo build --release && \
  strip ./target/release/rav1e && \
  mv ./target/release/rav1e /usr/local/bin 

RUN git clone https://github.com/gianni-rosato/svt-av1-psy && \
  cd svt-av1-psy/Build && \
  cmake .. -G"Unix Makefiles" -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release && \
  make -j$(nproc) install

RUN git clone --depth 1 https://aomedia.googlesource.com/aom && \
  cd aom && \
  mkdir build_tmp && cd build_tmp && cmake -DCMAKE_CXX_FLAGS="-O3 -march=native -pipe" -DBUILD_SHARED_LIBS=0 \
  -DENABLE_TESTS=0 -DENABLE_NASM=on -DCMAKE_INSTALL_LIBDIR=lib .. && make -j$(nproc) && make install

# pip
RUN MAKEFLAGS="-j$(nproc)" pip install timm wget cmake scipy meson ninja numpy einops kornia vsutil onnx

# deleting .so files to symlink them later on to save space
RUN pip install tensorrt==10.7.0 --pre tensorrt --extra-index-url https://pypi.nvidia.com/ && pip install polygraphy --extra-index-url https://pypi.nvidia.com/ && \
  rm -rf /root/.cache/ /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvinfer.so.* /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_builder_resource.so.* \
    /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_plugin.so.* /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvonnxparser.so.*

COPY --from=tensorrt-ubuntu /code/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-1.20.0-cp312-cp312-linux_x86_64.whl /workspace
RUN pip install coloredlogs flatbuffers numpy packaging protobuf sympy onnxruntime_gpu-1.20.0-cp312-cp312-linux_x86_64.whl

# vs_temporalfix dependencies
RUN apt install nasm libfftw3-dev -y && git clone https://github.com/dubhater/vapoursynth-mvtools && cd vapoursynth-mvtools && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/dubhater/vapoursynth-temporalmedian && cd vapoursynth-temporalmedian && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/dubhater/vapoursynth-motionmask && cd vapoursynth-motionmask && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/dubhater/vapoursynth-fillborders && cd vapoursynth-fillborders && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex && cd VapourSynth-Retinex && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny && cd VapourSynth-TCanny && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF && cd VapourSynth-CTMF && mkdir build && cd build && meson ../ && ninja && ninja install
RUN git clone https://github.com/vapoursynth/vs-removegrain && cd vs-removegrain && mkdir build && cd build && meson ../ && ninja && ninja install
# pifroggi plugins
RUN pip install git+https://github.com/pifroggi/vs_colorfix git+https://github.com/pifroggi/vs_temporalfix

# holywu
# todo: for now using official torch since torch_tensorrt is not compatible with my torch, but official whl has a bigger filesize, failed to compile torch_tensorrt
RUN pip install --pre -U torch torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124 --extra-index-url https://pypi.nvidia.com --no-deps && \
  pip install git+https://github.com/HolyWu/vs-rife --no-deps
# required for torch import
RUN wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz && \
  tar -xf libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz && cd libcusparse_lt-linux-x86_64-0.6.3.2-archive/lib && mv * /usr/local/lib && ldconfig
RUN python -m vsrife

# installing own versions
COPY --from=cupy-ubuntu /cupy/dist/ /workspace
# todo: for now using official torch since torch_tensorrt is not compatible with my torch, but official whl has a bigger filesize
#COPY --from=torch-ubuntu /pytorch/dist/ /workspace
RUN pip uninstall -y cupy* && \
  find . -name "*whl" ! -path "./Python-3.12.7/*" -exec pip install {} \;

# ddfi csv
RUN pip install pandas

# workaround for arch updates
# ffmpeg: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by ffmpeg)
# ffmpeg: /usr/lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.38' not found (required by ffmpeg)
# ffmpeg: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by ffmpeg)
RUN mkdir /workspace/hotfix
WORKDIR /workspace/hotfix
RUN wget https://mirrors.edge.kernel.org/ubuntu/pool/main/libt/libtirpc/libtirpc-dev_1.3.4%2Bds-1.3_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/libx/libxcrypt/libcrypt-dev_4.4.36-4build1_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/libx/libxcrypt/libcrypt1_4.4.36-4build1_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/libn/libnsl/libnsl-dev_1.3.0-3build3_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/g/glibc/libc6_2.40-1ubuntu3_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/g/glibc/libc6-dev_2.40-1ubuntu3_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/g/glibc/libc-bin_2.40-1ubuntu3_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/g/glibc/libc-dev-bin_2.40-1ubuntu3_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/l/linux/linux-libc-dev_6.11.0-12.13_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/r/rpcsvc-proto/rpcsvc-proto_1.4.2-0ubuntu7_amd64.deb \
    https://mirrors.edge.kernel.org/ubuntu/pool/main/libt/libtirpc/libtirpc3t64_1.3.4%2Bds-1.3_amd64.deb

###################
# final
############################
FROM nvidia/cuda:12.5.1-runtime-ubuntu24.04 AS final
# maybe official tensorrt image is better
#FROM nvcr.io/nvidia/tensorrt:23.04-py3 as final
ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

WORKDIR workspace

# install python
COPY --from=base /usr/local/bin/python /usr/local/bin/
COPY --from=base /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=base /workspace/Python-3.12.7/libpython3.12.so* /workspace/Python-3.12.7/libpython3.so \
  /workspace/Python-3.12.7/libpython3.so /usr/lib

# vapoursynth
COPY --from=base /workspace/zimg/zimg_0.0-1_amd64.deb zimg_0.0-1_amd64.deb
RUN apt install ./zimg_0.0-1_amd64.deb -y && rm -rf zimg_0.0-1_amd64.deb

COPY --from=base /usr/local/lib/vapoursynth /usr/local/lib/vapoursynth
COPY --from=base /usr/local/lib/x86_64-linux-gnu/vapoursynth /usr/local/lib/x86_64-linux-gnu/vapoursynth
COPY --from=base /usr/local/lib/libvapoursynth-script.so* /usr/local/lib/libvapoursynth.so /usr/local/lib/

# vapoursynth
COPY --from=base /usr/local/bin/vspipe  /usr/local/bin/vspipe

# vs plugins
COPY --from=base /usr/local/lib/libvstrt.so /usr/local/lib/libfmtconv.so /usr/local/lib/
COPY --from=base /usr/lib/x86_64-linux-gnu/libfftw3f.so* /usr/lib/x86_64-linux-gnu/

COPY --from=bestsource-lsmash-ffms2-vs /usr/local/lib/liblsmash.so* /usr/local/lib/
COPY --from=bestsource-lsmash-ffms2-vs /usr/local/lib/vapoursynth/libvslsmashsource.so* /usr/local/lib/vapoursynth/
COPY --from=bestsource-lsmash-ffms2-vs /usr/local/lib/vapoursynth/bestsource.so* /usr/local/lib/vapoursynth/bestsource.so
COPY --from=bestsource-lsmash-ffms2-vs /usr/local/lib/x86_64-linux-gnu/libbestsource.so* /usr/local/lib/x86_64-linux-gnu/libbestsource.so
COPY --from=bestsource-lsmash-ffms2-vs /usr/local/lib/libffms2.so*  /usr/local/lib/

COPY --from=base /usr/local/lib/vapoursynth/libvmaf.so /usr/local/lib/vapoursynth/libdescale.so /usr/local/lib/vapoursynth/libakarin.so \
  /usr/local/lib/vapoursynth/libmiscfilters.so /usr/local/lib/vapoursynth/libcas.so /usr/local/lib/vapoursynth/libremovegrain.so \
  /usr/local/lib/vapoursynth/libretinex.so /usr/local/lib/vapoursynth/libtcanny.so /usr/local/lib/vapoursynth/

COPY --from=base /usr/local/lib/x86_64-linux-gnu/vapoursynth/libvfrtocfr.so /usr/local/lib/x86_64-linux-gnu/libvmaf.so /usr/local/lib/x86_64-linux-gnu/vapoursynth/libvfrtocfr.so \
  /usr/local/lib/x86_64-linux-gnu/libvmaf.so /usr/local/lib/x86_64-linux-gnu/libawarpsharp2.so /usr/local/lib/x86_64-linux-gnu/libmvtools.so \
  /usr/local/lib/x86_64-linux-gnu/libfillborders.so /usr/local/lib/x86_64-linux-gnu/libmotionmask.so /usr/local/lib/x86_64-linux-gnu/libtemporalmedian.so /usr/local/lib/x86_64-linux-gnu/

# vsort
COPY --from=vsort-ubuntu /workdir/vs-mlrt/vsort/build/libvsort.so /usr/local/lib/
COPY --from=vsort-ubuntu /workdir/vs-mlrt/onnxruntime/lib/libonnxruntime.so /workdir/vs-mlrt/onnxruntime/lib/libonnxruntime_providers_shared.so \
  /workdir/vs-mlrt/onnxruntime/lib/libonnxruntime_providers_cuda.so /workdir/vs-mlrt/onnxruntime/lib/libonnxruntime_providers_tensorrt.so /usr/local/lib/

# av1an / rav1e / svt / aom
COPY --from=base /usr/bin/av1an /usr/local/bin/rav1e /usr/bin/
COPY --from=base /usr/local/bin/SvtAv1EncApp /usr/local/bin/aomenc /usr/local/bin/
# ffmpeg
COPY --from=ffmpeg-arch /home/makepkg/FFmpeg/ffmpeg /usr/local/bin/ffmpeg

# windows hotfix
RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvidia* /usr/lib/x86_64-linux-gnu/libcuda*

# trt
COPY --from=tensorrt-ubuntu /usr/local/tensorrt/lib/libnvinfer_plugin.so* /usr/local/tensorrt/lib/libnvinfer_vc_plugin.so* /usr/local/tensorrt/lib/libnvonnxparser.so* /usr/lib/x86_64-linux-gnu/
COPY --from=tensorrt-ubuntu /usr/local/cudnn/lib/libcudnn*.so* /usr/local/tensorrt/lib/libnvinfer.so* /usr/local/tensorrt/lib/libnvinfer_builder_resource.so* \
  /usr/local/tensorrt/lib/libnvonnxparser.so* /usr/local/tensorrt/lib/libnvparsers.so.8* /usr/local/tensorrt/lib/libnvinfer_plugin.so.8* /usr/lib/x86_64-linux-gnu/

# ffmpeg (todo: try to make it fully static)
COPY --from=base /usr/lib/x86_64-linux-gnu/libxcb*.so* /usr/lib/x86_64-linux-gnu/libgomp*.so* /usr/lib/x86_64-linux-gnu/libfontconfig.so* \
  /usr/lib/x86_64-linux-gnu/libfreetype.so* /usr/lib/x86_64-linux-gnu/libfribidi.so* /usr/lib/x86_64-linux-gnu/libharfbuzz.so* /usr/lib/x86_64-linux-gnu/libxml2.so* \
  /usr/lib/x86_64-linux-gnu/libsoxr.so* /usr/lib/x86_64-linux-gnu/libXau.so* /usr/lib/x86_64-linux-gnu/libXdmcp.so* \
  /usr/lib/x86_64-linux-gnu/libexpat.so* /usr/lib/x86_64-linux-gnu/libpng16.so* /usr/lib/x86_64-linux-gnu/libbrotlidec.so* /usr/lib/x86_64-linux-gnu/libglib-2.0.so* /usr/lib/x86_64-linux-gnu/libgraphite2.so* \
  /usr/lib/x86_64-linux-gnu/libicuuc.so* /usr/lib/x86_64-linux-gnu/libbsd.so* /usr/lib/x86_64-linux-gnu/libbrotlicommon.so* /usr/lib/x86_64-linux-gnu/libicudata.so* \
  /usr/lib/x86_64-linux-gnu/libicudata.so* /usr/lib/x86_64-linux-gnu/libmd.so* /usr/lib/x86_64-linux-gnu/libdrm.so* \
  /usr/lib/x86_64-linux-gnu/
COPY --from=ffmpeg-arch /usr/lib/libstdc++.so* /usr/lib/x86_64-linux-gnu/

# symlink python tensorrt
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvinfer.so.10
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource.so.10.7.0 /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_builder_resource.so.10.7.0
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10 /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_plugin.so.10
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 /usr/local/lib/python3.12/site-packages/tensorrt_libs/libnvonnxparser.so.10

# move trtexec so it can be globally accessed
COPY --from=tensorrt-ubuntu /usr/local/tensorrt/bin/trtexec /usr/bin

# torch
COPY --from=torch-ubuntu /usr/lib/x86_64-linux-gnu/libopenblas.so* /usr/lib/x86_64-linux-gnu/libgfortran.so* \
  /usr/lib/x86_64-linux-gnu/libquadmath.so* /usr/lib/x86_64-linux-gnu/
COPY --from=torch-ubuntu /usr/local/cuda-12.5/targets/x86_64-linux/lib/libcupti.so /usr/lib/x86_64-linux-gnu/

# ffmpeg hotfix
COPY --from=base /workspace/hotfix/* /workspace
RUN dpkg --force-all -i *.deb  && rm -rf *deb

RUN ldconfig

# fixing polygraphy
# AttributeError: `np.unicode_` was removed in the NumPy 2.0 release. Use `np.str_` instead.
RUN sed -i 's/np.unicode_/np.str_/g' /usr/local/lib/python3.12/site-packages/polygraphy/datatype/numpy.py

# pytorch import fix
COPY --from=base /usr/local/lib/libcusparseLt.so /usr/local/lib/libcusparseLt.so.0 /usr/local/lib/libcusparseLt.so.0.6.3.2 /usr/local/lib/

ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace/tensorrt
