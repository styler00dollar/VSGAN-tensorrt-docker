# Code mainly from https://github.com/HolyWu/vs-realesrgan
import vapoursynth as vs
import os
import numpy as np
from PIL import Image
from realsr_ncnn_vulkan_python import RealSR
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def SRVGGNetCompactRealESRGAN_ncnn(clip: vs.VideoNode, gpuid: int = 0, model: str = "models-DF2K", tta_mode: bool = False, scale: int = 2, tilesize: int = 0, param_path: str = None, bin_path: str = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if scale not in [2, 4]:
        raise vs.Error('RealESRGAN: scale must be 2 or 4')
    
    model = RealSR(gpuid=gpuid, model=model, tta_mode=tta_mode, scale=scale, tilesize=tilesize, param_path=param_path, bin_path=bin_path)

    def realesrgan(n, f):
        img = frame_to_tensor(f[0])
        output = model.process(img)
        return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_tensor(f):
    arr = np.stack([np.asarray(f.get_read_array(plane) if vs_api_below4 else f[plane]) for plane in range(f.format.num_planes)])
    # (3, 480, 848)
    arr = arr * 255
    arr = Image.fromarray(np.rollaxis(arr.clip(0,255).astype(np.uint8), 0,3))
    return arr


def tensor_to_frame(t, f):
    arr = np.array(t)
    # (960, 1696, 3)
    arr = arr.swapaxes(0, 2).swapaxes(1, 2)
    arr = arr / 255
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f.get_write_array(plane) if vs_api_below4 else f[plane]), arr[plane, :, :])
    return f
