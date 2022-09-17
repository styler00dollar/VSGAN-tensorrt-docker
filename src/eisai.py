import itertools
import numpy as np
import vapoursynth as vs
import functools
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def EISAI(
    clip: vs.VideoNode,
    scale: float = 1.0,
    fastmode: bool = False,
    ensemble: bool = False,
    skip_framelist=[],
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("EISAI: this is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("EISAI: only RGBS format is supported")

    if clip.num_frames < 2:
        raise vs.Error("EISAI: clip's number of frames must be at least 2")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error("EISAI: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    core = vs.core

    from .eisai_arch import SoftsplatLite, DTM, RAFT, interpolate
    import torch

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # load models
    device = torch.device("cuda")

    ssl = SoftsplatLite()
    dtm = DTM()
    ssl.load_state_dict(torch.load("/workspace/tensorrt/models//ssl.pt"))
    dtm.load_state_dict(torch.load("./workspace/tensorrt/models/dtm.pt"))
    ssl = ssl.to(device).eval()
    dtm = dtm.to(device).eval()
    raft = RAFT(path="/workspace/tensorrt/models/anime_interp_full.ckpt").eval().to(device)

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
        if (n % 2 == 0) or n == 0 or n in skip_framelist or n == clip.num_frames - 1:
            return clip

        I0 = frame_to_tensor(clip.get_frame(n - 1))
        I1 = frame_to_tensor(clip.get_frame(n + 1))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

        # clamping because vs does not give tensors in range 0-1, results in nan in output
        I0 = torch.clamp(I0, min=0, max=1)
        I1 = torch.clamp(I1, min=0, max=1)

        middle = interpolate(raft, ssl, dtm, I0, I1)

        middle = middle.detach().cpu().numpy()

        return tensor_to_clip(clip=clip, image=middle)

    clip = core.std.Interleave([clip, clip])
    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )
