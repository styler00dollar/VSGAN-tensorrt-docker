# https://github.com/AmusementClub/vs-mlrt/blob/7856cd3aab7111345475f7d526fe75ef81e198c8/scripts/vsmlrt.py

import vapoursynth as vs
from vapoursynth import core

def rife_trt(    
    clip: vs.VideoNode,
    multi: int = 2,
    scale: float = 1.0,
    device_id: int = 0,
    num_streams: int = 1,
    engine_path: str = ""):

    initial = core.std.Interleave([clip] * (multi - 1))

    terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    terminal = core.std.Interleave([terminal] * (multi - 1))

    timepoint = core.std.Interleave([
        clip.std.BlankClip(format=vs.GRAYS, color=i/multi, length=1)
        for i in range(1, multi)
    ]).std.Loop(clip.num_frames)

    scale = core.std.Interleave([
        clip.std.BlankClip(format=vs.GRAYS, color=scale, length=1)
        for i in range(1, multi)
    ]).std.Loop(clip.num_frames)

    clips = [initial, terminal, timepoint, scale]

    output = core.trt.Model(
        clips, engine_path,
        device_id=device_id,
        num_streams=num_streams
    )

    if multi == 2:
        return core.std.Interleave([clip, output])
    else:
        return core.std.Interleave([
            clip,
            *(output.std.SelectEvery(cycle=multi-1, offsets=i) for i in range(multi - 1))
        ])