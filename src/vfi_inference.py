import itertools
import numpy as np
import vapoursynth as vs
import functools
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .dedup import PSNR
import torch

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def vfi_inference(
    model_inference, clip: vs.VideoNode, skip_frame_list=[], multi=4
) -> vs.VideoNode:
    core = vs.core

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

    def execute(n: int, clip0: vs.VideoNode, clip1: vs.VideoNode) -> vs.VideoNode:
        if (
            (n % multi == 0)
            or n == 0
            or n in skip_frame_list
            or n // multi == clip.num_frames - 1
        ):
            return clip0

        I0 = frame_to_tensor(clip0.get_frame(n))
        I1 = frame_to_tensor(clip1.get_frame(n))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

        # clamping because vs does not give tensors in range 0-1, results in nan in output
        I0 = torch.clamp(I0, min=0, max=1)
        I1 = torch.clamp(I1, min=0, max=1)

        middle = model_inference.execute(I0, I1, (n % multi) / multi)

        return tensor_to_clip(clip=clip0, image=middle)

    clip0 = vs.core.std.Interleave([clip] * multi)
    clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.DeleteFrames(
        frames=0
    )
    clip1 = vs.core.std.Interleave([clip1] * multi)

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip0, width=clip.width, height=clip.height),
        functools.partial(execute, clip0=clip0, clip1=clip1),
    )


def vfi_frame_merger(
    clip1: vs.VideoNode,
    clip2: vs.VideoNode,
    skip_frame_list=[],
) -> vs.VideoNode:
    core = vs.core

    def execute(n: int, clip1: vs.VideoNode, clip2: vs.VideoNode) -> vs.VideoNode:
        if n in skip_frame_list:
            return clip1
        return clip2

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip1, width=clip1.width, height=clip1.height),
        functools.partial(execute, clip1=clip1, clip2=clip2),
    )
