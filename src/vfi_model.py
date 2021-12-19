import vapoursynth as vs
import torch
import numpy as np
import kornia
import os
from torch.nn import functional as F
import kornia
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def video_model(clip: vs.VideoNode, fp16: bool = False, model_path: str = "/workspace/rvpV1_105661_G.pt") -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('This is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('Only RGBS format is supported')

    if clip.num_frames < 2:
        raise vs.Error('Number of frames must be at least 2')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = torch.jit.load(model_path)
    model.eval()
    model.cuda()

    if fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model.half()

    w = clip.width
    h = clip.height

    clip0 = core.std.Interleave([clip, clip])
    clip1 = clip0.std.DuplicateFrames(frames=clip0.num_frames - 1).std.DeleteFrames(frames=0)

    @torch.inference_mode()
    def rife(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        if not (n & 1) or n == clip0.num_frames - 1 or f[0].props.get('_SceneChangeNext'):
            return f[0]

        I0 = frame_to_tensor(f[0]).to("cuda", non_blocking=True)
        I1 = frame_to_tensor(f[1]).to("cuda", non_blocking=True)

        I0 = kornia.color.yuv.rgb_to_yuv(I0)
        I1 = kornia.color.yuv.rgb_to_yuv(I1)

        if fp16:
            I0 = I0.half()
            I1 = I1.half()

        middle = model(I0, I1)

        middle = kornia.color.yuv.yuv_to_rgb(middle)

        return tensor_to_frame(middle[:, :, :h, :w], f[0])
    return clip0.std.ModifyFrame(clips=[clip0, clip1], selector=rife)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane) if vs_api_below4 else f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    fout = f.copy()
    for plane in range(fout.format.num_planes):
        np.copyto(np.asarray(fout.get_write_array(plane) if vs_api_below4 else fout[plane]), arr[plane, :, :])
    return fout