import itertools
import numpy as np
import vapoursynth as vs
import functools
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .dedup import PSNR
import torch

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def M2M(
    clip: vs.VideoNode,
    fp16: bool = False,
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("M2M: this is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("M2M: only RGBS format is supported")

    if clip.num_frames < 2:
        raise vs.Error("M2M: clip's number of frames must be at least 2")

    core = vs.core

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    from .M2M_arch import M2M_PWC

    model = M2M_PWC()
    model.load_state_dict(torch.load("/workspace/tensorrt/M2M.pth"))

    model.eval().cuda()

    w = clip.width
    h = clip.height

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
        if (n % 2 == 0) or n == 0 or n == clip.num_frames - 1:
            return clip

        # if frame number odd
        I0 = frame_to_tensor(clip.get_frame(n - 1))
        I1 = frame_to_tensor(clip.get_frame(n + 1))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

        if fp16:
            I0 = I0.half()
            I1 = I1.half()
        with torch.inference_mode():
            # forcing 2x for now
            intRatio = None
            intStep = 0.5
            tenSteps = torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()
            middle = model(I0, I1, tenSteps, intRatio)[0]

        middle = middle.detach().squeeze(0).cpu().numpy()

        return tensor_to_clip(clip=clip, image=middle)

    clip = core.std.Interleave([clip, clip])
    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )
