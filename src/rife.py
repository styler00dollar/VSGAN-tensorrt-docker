import itertools
import numpy as np
import vapoursynth as vs
import functools
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .dedup import PSNR

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
def RIFE(
    clip: vs.VideoNode,
    multi: int = 4,
    scale: float = 4.0,
    fp16: bool = True,
    fastmode: bool = False,
    ensemble: bool = True,
    psnr_dedup: bool = False,
    psnr_value: float = 70,
    ssim_dedup: bool = True,
    ms_ssim_dedup: bool = False,
    ssim_value: float = 0.999,
    skip_framelist=[],
    backend_inference: str = "cuda",
    model_version: str = "rife42",
) -> vs.VideoNode:
    """
    RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    In order to avoid artifacts at scene changes, you should invoke `misc.SCDetect` on YUV or Gray format of the input beforehand so as to set frame properties.

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        multi: Multiple of the frame counts.

        scale: Controls the process resolution for optical flow model. Try scale=0.5 for 4K video. Must be 0.25, 0.5, 1.0, 2.0, or 4.0.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("RIFE: this is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("RIFE: only RGBS format is supported")

    if clip.num_frames < 2:
        raise vs.Error("RIFE: clip's number of frames must be at least 2")

    if not isinstance(multi, int):
        raise vs.Error("RIFE: multi must be integer")

    if multi < 2:
        raise vs.Error("RIFE: multi must be at least 2")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error("RIFE: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    core = vs.core
    if backend_inference == "cuda":
        from .rife_arch import IFNet
        import torch

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        if model_version == "rife40":
            model = IFNet(arch_ver="4.0")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife40.pth"), False)
        elif model_version == "rife41":
            model = IFNet(arch_ver="4.0")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife41.pth"), False)
        elif model_version == "rife42":
            model = IFNet(arch_ver="4.2")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife42.pth"), False)
        elif model_version == "rife43":
            model = IFNet(arch_ver="4.3")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife43.pth"), False)
        elif model_version == "rife44":
            model = IFNet(arch_ver="4.3")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife44.pth"), False)
        elif model_version == "rife45":
            model = IFNet(arch_ver="4.5")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife45.pth"), False)
        elif model_version == "rife46":
            model = IFNet(arch_ver="4.6")
            model.load_state_dict(torch.load("/workspace/tensorrt/models/rife46.pth"), False)      
        elif model_version == "sudo_rife4":
            model.load_state_dict(
                torch.load("/workspace/tensorrt/models/sudo_rife4_269.662_testV1_scale1.pth"), False
            )

        model.eval().cuda()
    elif backend_inference == "ncnn":
        from rife_ncnn_vulkan_python import Rife

        model = Rife(gpuid=0, model="rife-v4", tta_mode=False, num_threads=4)

    w = clip.width
    h = clip.height
    scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]

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

            # if too similar
            if (
                PSNR(I0.swapaxes(0, 2).swapaxes(0, 1), I1.swapaxes(0, 2).swapaxes(0, 1))
                > psnr_value
                and psnr_dedup == True
            ):
                return clip

            if backend_inference == "cuda":
                I0 = torch.Tensor(I0).unsqueeze(0).to("cuda", non_blocking=True)
                I1 = torch.Tensor(I1).unsqueeze(0).to("cuda", non_blocking=True)

                # if too similar
                if (ssim(I0, I1) > ssim_value and ssim_dedup == True) or (
                    ms_ssim(I0, I1) > ssim_value and ms_ssim_dedup == True
                ):
                    return clip

                if fp16:
                    I0 = I0.half()
                    I1 = I1.half()

                middle = model(
                    I0, I1, scale_list=scale_list, fastmode=fastmode, ensemble=ensemble
                )
                middle = middle.detach().squeeze(0).cpu().numpy()

            elif backend_inference == "ncnn":
                I0 = I0.swapaxes(0, 2).swapaxes(0, 1) * 255
                I0 = np.clip(I0, 0, 255).astype(np.int8)
                I1 = I1.swapaxes(0, 2).swapaxes(0, 1) * 255
                I1 = np.clip(I1, 0, 255).astype(np.int8)
                middle = model.process(I0, I1)
                middle = middle.swapaxes(0, 2).swapaxes(1, 2).astype(np.float16) / 255

            return tensor_to_clip(clip=clip, image=middle)

        clip = core.std.Interleave([clip, clip])
        return core.std.FrameEval(
            core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
            functools.partial(execute, clip=clip),
        )

    else:

        @torch.inference_mode()
        def rife(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
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

            output = model(
                I0,
                I1,
                timestep=(n % multi) / multi,
                scale_list=scale_list,
                fastmode=fastmode,
                ensemble=ensemble,
            )

            return tensor_to_frame(output, f[0].copy())

        def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
            arr = np.stack(
                [np.asarray(f[plane]) for plane in range(f.format.num_planes)]
            )
            return torch.from_numpy(arr).unsqueeze(0)

        def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
            arr = t.squeeze(0).detach().cpu().numpy()
            for plane in range(f.format.num_planes):
                np.copyto(np.asarray(f[plane]), arr[plane, :, :])
            return f

        clip0 = vs.core.std.Interleave([clip] * multi)
        clip1 = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.DeleteFrames(
            frames=0
        )
        clip1 = vs.core.std.Interleave([clip1] * multi)
        return clip0.std.ModifyFrame(clips=[clip0, clip1], selector=rife)
