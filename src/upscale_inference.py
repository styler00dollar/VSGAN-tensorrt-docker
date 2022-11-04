import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch
from .realesrganner import RealESRGANer

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def upscale_inference(
    upscale_model_inference,
    clip: vs.VideoNode,
    tile_x=512,
    tile_y=512,
    tile_pad=10,
    pre_pad=0,
    ssim_value=0.999,
) -> vs.VideoNode:
    core = vs.core
    scale = upscale_model_inference.scale

    upsampler = RealESRGANer(
        scale,
        upscale_model_inference,
        tile_x=tile_x,
        tile_y=tile_y,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
    )

    def frame_to_tensor(frame: vs.VideoFrame):
        return np.stack(
            [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
        )

    def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
        for plane in range(f.format.num_planes):
            d = np.asarray(f[plane])
            np.copyto(d, array[plane, :, :])
        return f

    def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
        clip = core.std.BlankClip(
            clip=clip, width=image.shape[-1], height=image.shape[-2]
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: tensor_to_frame(f.copy(), image),
        )

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        clip_metric = clip.get_frame(n).props.get("float_ssim")
        if clip_metric and clip_metric > ssim_value:
            return clip

        I0 = frame_to_tensor(clip.get_frame(n))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)

        # clamping because vs does not give tensors in range 0-1, results in nan in output
        I0 = torch.clamp(I0, min=0, max=1)

        with torch.inference_mode():
            out = upsampler.enhance(I0)
            out = out.detach().squeeze().cpu().numpy()

        return tensor_to_clip(clip=clip, image=out)

    cache = {}
    interval = 5

    def execute_cache(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        clip_metric = clip.get_frame(n).props.get("float_ssim")
        if clip_metric and clip_metric > ssim_value:
            return clip

        if str(n) not in cache:
            cache.clear()
            # clamping because vs does not give tensors in range 0-1, results in nan in output
            images = [torch.Tensor(frame_to_tensor(clip.get_frame(n)))]
            for i in range(1, interval):
                if (n + i) >= clip.num_frames:
                    break
                images.append(torch.Tensor(frame_to_tensor(clip.get_frame(n + i))))

            images = torch.stack(images)
            images = images.unsqueeze(0)

            with torch.inference_mode():
                output = upscale_model_inference.execute(images)

            for i in range(output.shape[0]):
                cache[str(n + i)] = output[i, :, :, :].cpu().numpy()

            del output
            torch.cuda.empty_cache()
        return tensor_to_clip(clip=clip, image=cache[str(n)])

    if upscale_model_inference.cache:
        return core.std.FrameEval(
            core.std.BlankClip(
                clip=clip, width=clip.width * scale, height=clip.height * scale
            ),
            functools.partial(execute_cache, clip=clip),
        )
    return core.std.FrameEval(
        core.std.BlankClip(
            clip=clip, width=clip.width * scale, height=clip.height * scale
        ),
        functools.partial(execute, clip=clip),
    )


def upscale_frame_skip(clip: vs.VideoNode, ssim_value=0.999) -> vs.VideoNode:
    core = vs.core

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        if n == 0:
            return clip

        ssim_clip = clip.get_frame(n).props.get("float_ssim")
        if ssim_clip and ssim_clip > ssim_value:
            return clip[1::]
        return clip

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )
