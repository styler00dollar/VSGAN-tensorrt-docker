# VSGAN-tensorrt-docker

<!--ts-->
   * [Usage](#usage)
   * [Usage example](#usage-example)
   * [Individual examples](#individual-examples)
   * [vs-mlrt (C++ TRT)](#vs-mlrt)
   * [Shot Boundary Detection](#shot-boundry)
   * [multi-gpu](#multi-gpu)
   * [VFR (variable refresh rate)](#vfr)
   * [Benchmarks](#benchmarks)
   * [License](#license)
<!--te-->

-------

Currently working networks:
- Rife with [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife/), v2 with [WolframRhodium](https://github.com/WolframRhodium)
- Model based shot boundary detection with [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [wentaozhu/AutoShot](https://github.com/wentaozhu/AutoShot)
- [HolyWu/vs-realesrgan](https://github.com/HolyWu/vs-realesrgan)
- [Real-ESRGAN / SRVGGNetCompact](https://github.com/xinntao/Real-ESRGAN)
- [SAFMN](https://github.com/sunny2109/SAFMN)
- [DPIR](https://github.com/cszn/DPIR)
- [Waifu2x](https://github.com/AmusementClub/vs-mlrt/releases/download/model-20211209/waifu2x_v3.7z)
- [real-cugan](https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD)
- [apisr](https://github.com/kiteretsu77/apisr)
- [AnimeJaNai](https://github.com/the-database/mpv-upscale-2x_animejanai)
- [ModernSpanimation](https://github.com/TNTwise/Models/releases/tag/2x_ModernSpanimationV1)
- [AniScale](https://github.com/Sirosky/Upscale-Hub)
- [Anime1080Fixer by zarxrax](https://mega.nz/file/EAs2XZwA#-YrO3QSP8Waz2l5JVtrSew2Dk4gbT0xE0uA5hc5CYRo)
- [Adore](https://github.com/renarchi/Re-SISR/releases/tag/Adore)

Onnx files can be found [here](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models).

Also used:
- TensorRT C++ inference and python script usage with [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- bestsource with [vapoursynth/bestsource](https://github.com/vapoursynth/bestsource)
- trt precision check and upscale frame skip with [mafiosnik777/enhancr](https://github.com/mafiosnik777/enhancr)
- temporal fix with [pifroggi/vs_temporalfix](https://github.com/pifroggi/vs_temporalfix)
- color fix with [pifroggi/vs_colorfix](https://github.com/pifroggi/vs_colorfix)

<div id='usage'/>

## Usage
The following docker requires the latest Nvidia driver (560+). After that, follow the following steps:

**WARNING FOR PEOPLE WITHOUT `AVX512`: Instead of using `styler00dollar/vsgan_tensorrt:latest`, which I build with my 7950x and thus with all AVX, use `styler00dollar/vsgan_tensorrt:latest_no_avx512` in `compose.yaml` to avoid `Illegal instruction (core dumped)` which is mentioned in [this issue](https://github.com/styler00dollar/VSGAN-tensorrt-docker/issues/48).**

**AND AS A FINAL INFO, `Error opening input file pipe:` IS NOT A REAL ERROR MESSAGE. That means invalid data got piped into ffmpeg and can be piped error messages for example. To see the actual error messages and what got piped, you can use `vspipe -c y4m inference.py -`.**

Quickstart:
```bash
# if you have Windows, install Docker Desktop https://www.docker.com/products/docker-desktop/

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
- `minimal`: Only has `ffmpeg`, `mlrt`, `ffms2`, `lsmash` and `bestsource`.

| docker image                                          | dockerhub | short description |
| ----------------------------------------------------- | --------- | ----------------- |
| styler00dollar/vsgan_tensorrt:latest                  | 13.1gb    | trt 10.16
| styler00dollar/vsgan_tensorrt:latest_no_avx512        | 13.1gb    | trt 10.16 without avx512
| styler00dollar/vsgan_tensorrt:minimal                 | 6.1gb     | trt 10.16 + ffmpeg + mlrt + ffms2 + lsmash + bestsource
| styler00dollar/vsgan_tensorrt:minimal_no_avx512       | 6.1gb     | trt 10.16 without avx512 + ffmpeg + mlrt + ffms2 + lsmash + bestsource

Piping usage:
```
vspipe -c y4m inference.py - | ffmpeg -i pipe: example.mkv -y
```
If docker does not want to start, try this before you use docker:
```bash
sudo systemctl start docker
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
DOCKER_BUILDKIT=1 sudo docker build -t styler00dollar/vsgan_tensorrt:latest .
# If you want to rebuild from scratch or have errors, try to build without cache
DOCKER_BUILDKIT=1 sudo docker build --no-cache -t styler00dollar/vsgan_tensorrt:latest .
```
Manually run docker:
```
# you need to be inside the vsgan folder with cli before running the following step, git clone repo and cd into it
# the folderpath before ":" will be mounted in the path which follows afterwards
# contents of the vsgan folder should appear inside /workspace/tensorrt

sudo docker run --privileged --gpus all -it --rm -v /home/vsgan_path/:/workspace/tensorrt styler00dollar/vsgan_tensorrt:latest

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
from vsrife import rife

sys.path.append("/workspace/tensorrt/")
core = vs.core
core.num_threads = 4


def inference_clip(video_path):
    clip = core.bs.VideoSource(source=video_path)

    clip = core.resize.Bicubic(
        clip, format=vs.RGBS, matrix_in_s="709"
    )  # RGBS means fp32, RGBH means fp16

    # interpolation
    clip = rife(clip, trt=True, model="4.26.heavy", sc=False)

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

<div id='individual-examples'/>

## Individual examples

More parameter documentation can be found in the plugin repositories.

- Video Reader: [ffms2](https://github.com/FFMS/ffms2/blob/master/doc/ffms2-vapoursynth.md) [lsmash](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/blob/master/VapourSynth/README.md) [bestsource](https://github.com/vapoursynth/bestsource)
```python
clip = core.ffms2.Source(source=video_path)
clip = core.bs.VideoSource(source=video_path) # recommended
```

- Convert colorspace: [descale](https://github.com/Irrational-Encoding-Wizardry/descale) [vs](http://www.vapoursynth.com/doc/functions/video/resize.html)
```python
clip = core.descale.Debilinear(clip, 1280, 720)
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s="709")
clip = core.resize.Bicubic(clip, width=1280, height=720,, format=vs.RGBS, matrix_in_s="709")
```

- Clamp 0-1
```python
clip = core.akarin.Expr(clip, "x 0 1 clamp")
clip = clip.std.Expr("x 0 max 1 min")
clip = core.std.Limiter(clip, max=1, planes=[0,1,2])
```

- Scene Detect
```python
from src.scene_detect import scene_detect
clip = scene_detect(clip, fp16=True, thresh=0.85, model=4)
```

- TensorRT inference: [vstrt](https://github.com/AmusementClub/vs-mlrt/tree/master/vstrt)
```python
clip = core.trt.Model(
    clip,
    engine_path="/workspace/tensorrt/cugan.engine",
    tilesize=[854, 480],
    overlap=[0, 0],
    num_streams=4,
)
```

- dpir inference: [vstrt](https://github.com/AmusementClub/vs-mlrt/tree/master/vstrt)
```python
strength = 10.0
noise_level = clip.std.BlankClip(format=vs.GRAYS, color=strength / 100)
clip = core.trt.Model(
    [clip, noise_level],
    engine_path="dpir.engine",
    num_streams=2,
)
```

- Rife: [vs-rife](https://github.com/styler00dollar/vs-rife)
```python
from vsrife import rife
clip = rife(clip, model="4.26.heavy", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")
```

- Compact/ESRGAN: [vs-realesrgan](https://github.com/HolyWu/vs-realesrgan)
```python
from vsrealesrgan import realesrgan
clip = realesrgan(clip, num_streams=3, batch_size=2, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt") 
```

- Sharpening: [awarp](https://github.com/HolyWu/VapourSynth-AWarp) [cas](https://github.com/HolyWu/VapourSynth-CAS)
```python
mask = core.edgemasks.Sobel(clip)
clip = core.awarp.AWarp(clip, mask, depth_h=8, depth_v=8, planes=[0,1,2])

clip = core.cas.CAS(clip, sharpness=0.5)
```

- Color fix: [colorfix](https://github.com/pifroggi/vs_colorfix)
```python
import vs_colorfix
clip = vs_colorfix.average(clip, ref, radius=10, planes=[0, 1, 2], fast=False)
```

- Temporal fix: [temporalfix](https://github.com/pifroggi/vs_temporalfix)
```python
import vs_temporalfix
clip = vs_temporalfix.model(clip, strength=2, tiles=1, backend="tensorrt", num_streams=3, engine_folder="/workspace/tensorrt/", exclude=None)
```

- Line darken: [linedarken](https://github.com/Selur/VapoursynthScriptsInHybrid/blob/550cc72221848732fa36a5dfb9ad5f98a308dd6e/havsfunc.py#L5217)
```python
from src.utils import FastLineDarkenMOD
clip = FastLineDarkenMOD(clip)
```

<div id='vs-mlrt'/>
  
## vs-mlrt (C++ TRT)
You need to convert onnx models into engines. You need to do that on the same system where you want to do inference. Download onnx models from [here]( https://github.com/AmusementClub/vs-mlrt/releases/download/v7/models.v7.7z) or from [my Github page](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/tag/models). Inside the docker, you do one of the following commands:
  
Good default choice:
```
trtexec --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5
```
If you have the vram to fit the model multiple times, add `--infStreams`.
```
trtexec --onnx=model.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=model.engine --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=4
```

**Warning:** Rife with TensorRT is broken without workarounds in every implementation (mlrt, torch_tensorrt, onnxruntime trt,...), even with fp32, and results in wobbly lines and artefacts during panning scenes. [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) fixed it [here](https://github.com/HolyWu/vs-rife/commit/0168d7e2ba5e362f234205e4118f15c5503b64f0) or [here](https://github.com/HolyWu/vs-rife/commit/4d6ff753b98cbbbb1a302c19f71315df7ac4da47). Testing showed that vsrife is not a lot slower than mlrt, so I don't recommend using rife onnx.

**Warnings**:
- Cugan with 3x scale requires same MIN/OPT/MAX shapes.
- If you use the FP16 onnx you need to use `RGBH` colorspace, if you use FP32 onnx you need to use `RGBS` colorspace in `inference_config.py` .
- Engines are system specific, don't use across multiple systems.
- Don't use reuse engines for different GPUs.
- If you run out of memory, then you need to adjust the resolutions in that command. If your video is bigger than what you can input in the command, use tiling.
- If you get segfault, reduce `builderOptimizationLevel`. Change can change it down to 1 to speed up the engine building, but may result in worse speeds.
- If you set min, opt and max to the same resolution, it might result in a faster engine.


<div id='shot-boundry'/>

## Shot Boundary Detection

`vsrife` can have scene detection if you set `sc`.

```python
clip = rife(clip, trt=True, model="4.26.heavy", sc=True)
```

Or use models. Adjust thresh to a value between 0 and 1, higher means to ignore with less confidence.

Example usage:
```python
from src.scene_detect import scene_detect
from vsrife import rife


clip_sc = scene_detect(
    clip,
    fp16=True,
    thresh=0.85,
    model=4,
)

clip = rife(clip, trt=True, model="4.26.heavy", sc=False)

clip_orig = core.std.Interleave([clip_orig] * 2)  # 2 means interpolation factor here
clip = core.akarin.Select([clip, clip_orig], clip_sc, "x._SceneChangeNext 1 0 ?")
```

Keep in mind that different models require a different thresh.

<div id='multi-gpu'/>

### multi-gpu

Thanks to tepete who figured it out, there is also a way to do inference on multipe GPUs.

```python
stream0 = core.std.SelectEvery(core.trt.Model(clip, engine_path="models/engines/model.engine", num_streams=2, device_id=0), cycle=3, offsets=0)
stream1 = core.std.SelectEvery(core.trt.Model(clip, engine_path="models/engines/model.engine", num_streams=2, device_id=1), cycle=3, offsets=1)
stream2 = core.std.SelectEvery(core.trt.Model(clip, engine_path="models/engines/model.engine", num_streams=2, device_id=2), cycle=3, offsets=2)
clip = core.std.Interleave([stream0, stream1, stream2])
```

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

<div id='benchmarks'/>

## Benchmarks

- Used `vspipe -c y4m inference.py - | ffmpeg -i pipe: -f null /dev/null -y` and a 5000 frame h264 default settings ffmpeg encoded video for all benchmarks.

| model                  | scale | os    | hardware     | arch                                                                 | fps 720 | fps 1080 | vram 720 | vram 1080   | backend                                                                           | verified output                    | batch | level | streams   | threads   | onnx      | onnxslim / onnxsim | onnx shape  | trtexec shape | precision | usage                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------- | ----- | ----- |------------- | -------------------------------------------------------------------- | ------- | -------- | -------- | ----------- | --------------------------------------------------------------------------------- | ---------------------------------- | ----- | ----- | --------- | --------- | --------- | ------------------ | ----------- | ------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AnimeJaNai V2          | 2x    | Linux | 4090 / 7950x | [compact](https://github.com/xinntao/Real-ESRGAN)                    | 139.70  | 61.52    | 3.2gb    | 5.8gb       | vs-realesrgan (fd23430), trt 10.9, torch2.6 (1eba9b3), torch_tensorrt2.6.0+cu126  | yes, works                         | 2     | 5     | 3         | 16        | -         | -		           | -           | static        | RGBH      | realesrgan(clip, num_streams=3, batch_size=2, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                 |
| ModernSpanimation V2   | 2x    | Linux | 4090 / 7950x | [span](https://github.com/hongyuanyu/SPAN)                           | 112.51  | 51.34    | 3.3gb    | 6.3gb       | trt10.9 (trtexec+mlrt)                                                            | yes, works                         | 1     | 5     | 3         | 16        | fp16 op20 | onnxslim           | dynamic     | dynamic       | RGBH      | trtexec --bf16 --fp16 --onnx=2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3         |
| cugan                  | 2x    | Linux | 4090 / 7950x | [cugan](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)      | 48.69   | 21.67    | 6.3gb    | 12.7gb      | trt10.9 (trtexec+mlrt)                                                            | yes, works                         | 1     | 5     | 3         | 16        | fp16 op20 | -                  | dynamic     | dynamic       | RGBH      | trtexec --bf16 --fp16 --onnx=cugan_pro-denoise3x-up2x_op18_fp16_clamp_colorfix.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=cugan_pro-denoise3x-up2x_op18_fp16_clamp_colorfix.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3     |
| vsrife 4.6             | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.6)                | 1022.81 | 426.79   | 1.7gb    | 2.1gb       | vs-rife (2a7f2c8), trt 10.9, torch2.6+12.6 (1eba9b3), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | rife(clip, model="4.6", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                             |
| vsrife 4.6 drba        | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.6)                | 549.47  | 303.41   | 1.8gb    | 2.2gb       | vs-drba (2a7f2c8), trt 10.9, torch2.6+12.6 (9200ded), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | drba_rife(clip, model="4.6", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                        |
| vsrife 4.7             | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.7)                | 1084.70 | 476.60   | 1.8gb    | 2.2gb       | vs-rife (2a7f2c8), trt 10.9, torch2.6+12.6 (1eba9b3), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | rife(clip, model="4.7", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                             |
| vsrife 4.7 drba        | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.7)                | 558.40  | 313.70   | 1.9gb    | 2.3gb       | vs-drba (2a7f2c8), trt 10.9, torch2.6+12.6 (9200ded), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | drba_rife(clip, model="4.7", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                        |
| vsrife 4.18            | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.18)               | 932.79  | 407.42   | 2.0gb    | 2.6gb       | vs-rife (2a7f2c8), trt 10.9, torch2.6+12.6 (1eba9b3), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | rife(clip, model="4.18", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                            |
| vsrife 4.18 drba       | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.18)               | 533.07  | 290.43   | 2.0gb    | 2.7gb       | vs-drba (2a7f2c8), trt 10.9, torch2.6+12.6 (9200ded), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | drba_rife(clip, model="4.18", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                       |
| vsrife 4.22            | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.22)               | 742.88  | 344.31   | 2.0gb    | 2.7gb       | vs-rife (2a7f2c8), trt 10.9, torch2.6+12.6 (1eba9b3), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | rife(clip, model="4.22", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                            |
| vsrife 4.22 drba       | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.22)               | 496.38  | 267.60   | 2.1gb    | 2.8gb       | vs-drba (2a7f2c8), trt 10.9, torch2.6+12.6 (9200ded), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | drba_rife(clip, model="4.22", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                       |
| vsrife 4.26            | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.22)               | 828.65  | 409.53   | 1.9gb    | 2.3gb       | vs-rife (2a7f2c8), trt 10.9, torch2.6+12.6 (1eba9b3), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | rife(clip, model="4.26", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                            |
| vsrife 4.26 drba       | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.26)               | 545.49  | 308.84   | 1.9gb    | 2.4gb       | vs-drba (2a7f2c8), trt 10.9, torch2.6+12.6 (9200ded), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | drba_rife(clip, model="4.26", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                       |
| vsrife 4.26.heavy      | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.26.heavy)         | 567.59  | 278.55   | 2.4gb    | 3.5gb       | vs-rife (2a7f2c8), trt 10.9, torch2.6+12.6 (1eba9b3), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | rife(clip, model="4.26.heavy", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                      |
| vsrife 4.26.heavy drba | 2x    | Linux | 4090 / 7950x | [rife](https://github.com/hzwer/Practical-RIFE) (4.26.heavy)         | 474.09  | 264.02   | 2.5gb    | 3.6gb       | vs-drba (2a7f2c8), trt 10.9, torch2.6+12.6 (9200ded), torch_tensorrt2.6.0+cu126   | no                                 | 1     | 5     | -         | 16        | -         | -                  | -           | static        | RGBH      | drba_rife(clip, model="4.26.heavy", sc=False, trt=True, trt_static_shape=True, trt_optimization_level=5, trt_cache_dir="/workspace/tensorrt")                                                                                                                                                                                                                                                 |

| model                 | scale | os             | hardware         | arch                                                                 | fps 720 | fps 1080 | vram 720 | vram 1080   | backend                                                                  | verified output  | batch | level | streams   | threads   | onnx      | onnxslim / onnxsim | onnx shape  | trtexec shape | precision | usage                                                                                                                                                                                                                                                                                                                                                                                         |
| --------------------- | ----- | -------------- | ---------------- | -------------------------------------------------------------------- | ------- | -------- | -------- | ----------- | ------------------------------------------------------------------------ | ---------------- | ----- | ----- | --------- | --------- | --------- | ------------------ | ----------- | ------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AnimeJaNai V2         | 2x    | Windows 11 Pro | 5090 / i9-13900k | [compact](https://github.com/xinntao/Real-ESRGAN)                    | 127.59  | 55.79    | 3.1gb    | 4.1gb       | trt 10.8 (trtexec+mlrt)                                                  | no               | 1     | 5     | 3         | 4         | fp16 op18 | -		            | dynamic     | dynamic       | RGBH      | trtexec --bf16 --fp16 --onnx=2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3               |
| AnimeJaNai V2         | 2x    | Windows 11 Pro | 5090 / i9-13900k | [compact](https://github.com/xinntao/Real-ESRGAN)                    | 167.37  | 74.85    | 7.7gb    | 14.5gb      | trt 10.8 (trtexec+mlrt)                                                  | no               | 2     | 5     | 10        | 10        | fp16 op18 | -		            | dynamic     | dynamic       | RGBH      | trtexec --bf16 --fp16 --onnx=2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp_batch2.onnx --minShapes=input:1x6x8x8 --optShapes=input:1x6x720x1280 --maxShapes=input:1x6x1080x1920 --saveEngine=2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp_batch2.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3 |
| ModernSpanimation V2  | 2x    | Windows 11 Pro | 5090 / i9-13900k | [span](https://github.com/hongyuanyu/SPAN)                           | 138.76  | 64.33    | 4.7gb    | 7.6gb       | trt 10.8 (trtexec+mlrt)                                                  | no               | 1     | 5     | 3         | 4         | fp16 op20 | onnxslim           | dynamic     | dynamic       | RGBH      | trtexec --bf16 --fp16 --onnx=2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=2x_ModernSpanimationV2_clamp_op20_fp16_onnxslim.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3         |
| cugan                 | 2x    | Windows 11 Pro | 5090 / i9-13900k | [cugan](https://github.com/bilibili/ailab/tree/main/Real-CUGAN)      | 71.21   | 31.22    | 7.3gb    | 13.4gb      | trt 10.8 (trtexec+mlrt)                                                  | no               | 1     | 5     | 3         | 4         | fp16 op20 | -                  | dynamic     | dynamic       | RGBH      | trtexec --bf16 --fp16 --onnx=cugan_pro-denoise3x-up2x_op18_fp16_clamp_colorfix.onnx --minShapes=input:1x3x8x8 --optShapes=input:1x3x720x1280 --maxShapes=input:1x3x1080x1920 --saveEngine=cugan_pro-denoise3x-up2x_op18_fp16_clamp_colorfix.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3     |
| dpir                  | 1x    | Windows 11 Pro | 5090 / i9-13900k | [dpir](https://github.com/cszn/DPIR) (4ch)                           | 63.03   | 28.89    | 4.1gb    | 6.2gb       | trt 10.8 (trtexec+mlrt)                                                  | no               | 1     | 5     | 3         | 4         | fp32 op9  | -                  | dynamic     | dynamic       | RGBS      | trtexec --bf16 --fp16 --onnx=dpir_drunet_color.onnx --minShapes=input:1x4x8x8 --optShapes=input:1x4x720x1280 --maxShapes=input:1x4x1080x1920 --saveEngine=dpir_drunet_color.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT --skipInference --useCudaGraph --noDataTransfers --builderOptimizationLevel=5 --infStreams=3                                                                     |

Other benchmarks that are not using the gpu:

|                                                                    | cpu    | fps 720 | fps 1080 | fps 4k | threads | cache | usage                                                                             |
| ------------------------------------------------------------------ | ------ | ------- | -------- | ------ | ------- | ----- | --------------------------------------------------------------------------------- |
| [vs_temporalfix](https://github.com/pifroggi/vs_temporalfix)         | 7950x  | 18.59   | 7.59     | 3.69   | 32      | 40000 | vs_temporalfix(clip, strength=400, tr=6, exclude="[10 20]", debug=False)          |

<div id='license'/>

## License

This code uses code from other repositories, but the code I wrote myself is under BSD3.


