import itertools
import numpy as np
import vapoursynth as vs
import functools
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .dedup import PSNR
import torch

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def M2M(clip: vs.VideoNode, fp16: bool = False, multi=4) -> vs.VideoNode:
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
    model.load_state_dict(torch.load("/workspace/tensorrt/models/M2M.pth"))

    model.eval().cuda()

    w = clip.width
    h = clip.height

    # using frameeval if multi = 2
    if multi == 2:

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

    else:
        cache = {}

        @torch.inference_mode()
        def m2m(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            if str(n) not in cache:
                cache.clear()

                if (
                    (n % multi == 0)
                    or (n // multi == clip.num_frames - 1)
                    or f[0].props.get("_SceneChangeNext")
                ):
                    return f[0]

                I0 = frame_to_tensor(f[0]).to("cuda", non_blocking=True)
                I1 = frame_to_tensor(f[1]).to("cuda", non_blocking=True)

                if fp16:
                    I0 = I0.half()
                    I1 = I1.half()

                intRatio = multi
                intStep = multi - 1
                tenSteps = [
                    torch.FloatTensor([st / intStep * 1]).view(1, 1, 1, 1).cuda()
                    for st in range(0, intStep)
                ]

                output = model(I0, I1, tenSteps, intRatio)
                output = torch.cat(output)

                for i in range(output.shape[0]):
                    cache[str(n + i)] = output[i, :, :, :]

                del output
                torch.cuda.empty_cache()

            return tensor_to_frame(cache[str(n)], f[0].copy())

        def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
            arr = np.stack(
                [np.asarray(f[plane]) for plane in range(f.format.num_planes)]
            )
            return torch.from_numpy(arr).unsqueeze(0)

        def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
            arr = t.detach().cpu().numpy()
            for plane in range(f.format.num_planes):
                np.copyto(np.asarray(f[plane]), arr[plane, :, :])
            return f

        clip0 = vs.core.std.Interleave([clip] * multi)
        clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.DeleteFrames(
            frames=0
        )
        clip1 = vs.core.std.Interleave([clip1] * multi)
        return clip0.std.ModifyFrame(clips=[clip0, clip1], selector=m2m)
