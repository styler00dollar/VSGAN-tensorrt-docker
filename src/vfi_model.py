import vapoursynth as vs
import torch
import numpy as np
import kornia
from torch.nn import functional as F
import kornia
import functools

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4


def video_model(
    clip: vs.VideoNode,
    fp16: bool = False,
    model_path: str = "/workspace/rvpV1_105661_G.pt",
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("This is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("Only RGBS format is supported")

    if clip.num_frames < 2:
        raise vs.Error("Number of frames must be at least 2")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = torch.jit.load(model_path)
    model.eval()
    model.cuda()

    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model.half()

    @torch.inference_mode()
    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        if (n % 2 == 0) or n == 0:
            return clip

        # if frame number odd
        I0 = frame_to_tensor(clip.get_frame(n - 1))
        I1 = frame_to_tensor(clip.get_frame(n + 1))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

        I0 = kornia.color.yuv.rgb_to_yuv(I0)
        I1 = kornia.color.yuv.rgb_to_yuv(I1)

        if fp16:
            I0 = I0.half()
            I1 = I1.half()

        middle = model(I0, I1)

        middle = kornia.color.yuv.yuv_to_rgb(middle)
        middle = middle.detach().squeeze(0).cpu().numpy()

        return tensor_to_clip(clip=clip, image=middle)

    clip = core.std.Interleave([clip, clip])
    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )


def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    return np.stack(
        [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
    )


def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f


def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(clip=clip, width=image.shape[-1], height=image.shape[-2])
    return core.std.ModifyFrame(
        clip=clip, clips=clip, selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )
