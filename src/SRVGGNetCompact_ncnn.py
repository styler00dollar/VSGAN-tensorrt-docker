# Code mainly from https://github.com/HolyWu/vs-realesrgan
import vapoursynth as vs
import os
import numpy as np
from PIL import Image
from realsr_ncnn_vulkan_python import RealSR
import functools
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

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        img = frame_to_tensor(clip.get_frame(n))

        img = img * 255
        img = np.rollaxis(img.clip(0,255).astype(np.uint8), 0,3)
        output = model.process(img)
        output = output.swapaxes(0, 2).swapaxes(1, 2)/255

        return tensor_to_clip(clip=clip, image=output)

    return core.std.FrameEval(
            core.std.BlankClip(
                clip=clip,
                width=clip.width * scale,
                height=clip.height * scale
            ),
            functools.partial(
                execute,
                clip=clip
            )
    )

def frame_to_tensor(frame: vs.VideoFrame):
    return np.stack([
        np.asarray(frame[plane])
        for plane in range(frame.format.num_planes)
    ])

def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f

def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(
        clip=clip,
        width=image.shape[-1],
        height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )
