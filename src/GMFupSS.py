import itertools
import numpy as np
import vapoursynth as vs
import functools

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def GMFupSS(
    clip: vs.VideoNode,
    scale: float = 4.0,
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("GMFupSS: this is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("GMFupSS: only RGBS format is supported")

    if clip.num_frames < 2:
        raise vs.Error("GMFupSS: clip's number of frames must be at least 2")

    if not isinstance(multi, int):
        raise vs.Error("GMFupSS: multi must be integer")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error("GMFupSS: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    core = vs.core
    if backend_inference == "cuda":
        from .GMFupSS_arch import Model_inference
        import torch

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = Model_inference()
    model.eval()

    w = clip.width
    h = clip.height
    scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]


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
        if (
            (n % 2 == 0)
            or n == 0
            or n in skip_framelist
            or n == clip.num_frames - 1
        ):
            return clip

        # if frame number odd
        I0 = frame_to_tensor(clip.get_frame(n - 1))
        I1 = frame_to_tensor(clip.get_frame(n + 1))

        I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
        I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)
        with torch.inference_mode():
            middle = model(I0, I1).cpu()
        return tensor_to_clip(clip=clip, image=middle)

    clip = core.std.Interleave([clip, clip])
    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )