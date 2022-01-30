# https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt
FROM nvcr.io/nvidia/tensorrt:22.01-py3
ARG DEBIAN_FRONTEND=noninteractive
# if you have 404 problems when you build the docker, try to run the upgrade
#RUN apt-get dist-upgrade -y
RUN apt-get -y update
# torch
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install https://github.com/NVIDIA/Torch-TensorRT/releases/download/v1.0.0/torch_tensorrt-1.0.0-cp38-cp38-linux_x86_64.whl

# installing vapoursynth
RUN apt install ffmpeg autoconf libtool yasm python3.9 python3.9-venv python3.9-dev ffmsindex libffms2-4 libffms2-dev -y
RUN git clone https://github.com/sekrit-twc/zimg.git && cd zimg && ./autogen.sh && ./configure && make -j4 && make install && cd .. && rm -rf zimg
RUN pip install Cython
RUN git clone https://github.com/vapoursynth/vapoursynth.git && cd vapoursynth && ./autogen.sh && ./configure && make && make install && cd .. && ldconfig
RUN ln -s /usr/local/lib/python3.9/site-packages/vapoursynth.so /usr/lib/python3.9/lib-dynload/vapoursynth.so
RUN pip install vapoursynth

# onnx
RUN pip install onnx onnxruntime onnxruntime-gpu

# installing onnx tensorrt with a workaround, error with import otherwise
# https://github.com/onnx/onnx-tensorrt/issues/643
RUN git clone --depth 1 --branch 21.02 \
    https://github.com/onnx/onnx-tensorrt.git && \
    cd onnx-tensorrt && \
    cp -r onnx_tensorrt /usr/local/lib/python3.8/dist-packages && \
    cd .. && \
    rm -rf onnx-tensorrt

# downloading models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
# fatal anime
RUN wget https://de-next.owncube.com/index.php/s/x99pKzS7TNaErrC/download -O 4x_fatal_Anime_500000_G.pth
# rvp1
RUN pip install gdown && gdown --id 1IJe6WLvT43iwl-3J6ectgnjas5mjnQ51
# sepconv
RUN pip install cupy-cuda115
RUN wget http://content.sniklaus.com/resepconv/network-paper.pytorch -O sepconv.pth
# EGVSR
RUN wget https://github.com/Thmen/EGVSR/raw/master/pretrained_models/EGVSR_iter420000.pth
# rife4 (fixed rife4.0 model)
RUN gdown --id 1UzCbpjxWJsfiDjoc7wuzq3K0RCf5Oxr3
# RealBasicVSR_x4
RUN gdown --id 1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID

# optional, rvp uses it to convert colorspace
RUN pip install kornia
# image read/write for image inference
RUN pip install opencv-python

# vs plugings from others
# https://github.com/HolyWu/vs-swinir
RUN pip install --upgrade vsswinir && python -m vsswinir
# https://github.com/HolyWu/vs-basicvsrpp
RUN pip install --upgrade vsbasicvsrpp && python -m vsbasicvsrpp

# dependencies for RealBasicVSR_x4
# mmedit
RUN git clone https://github.com/open-mmlab/mmediting.git && cd mmediting && pip install -v -e .
# RealBasicVSR_x4 will download this
RUN wget "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" -P /root/.cache/torch/hub/checkpoints/
