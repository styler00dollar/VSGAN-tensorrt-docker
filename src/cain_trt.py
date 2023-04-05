# https://github.com/AmusementClub/vs-mlrt/blob/7856cd3aab7111345475f7d526fe75ef81e198c8/scripts/vsmlrt.py

import vapoursynth as vs
from vapoursynth import core
from .vfi_inference import vfi_frame_merger


def cain_trt(
    clip: vs.VideoNode,
    device_id: int = 0,
    num_streams: int = 1,
    engine_path: str = "",
):
    initial = core.std.Interleave([clip] * (2 - 1))

    terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    terminal = core.std.Interleave([terminal] * (2 - 1))

    clips = [initial, terminal]

    output = core.trt.Model(
        clips, engine_path, device_id=device_id, num_streams=num_streams
    )

    # using FrameEval is much faster than calling core.akarin.Expr
    output = vfi_frame_merger(initial, output)

    return core.std.Interleave([clip, output])
