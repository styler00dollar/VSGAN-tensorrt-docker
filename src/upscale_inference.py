import itertools
import numpy as np
import vapoursynth as vs
import functools

def upscale_frame_skip(
    clip: vs.VideoNode,
    skip_framelist=[],
) -> vs.VideoNode:
    core = vs.core

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        if n == 0:
            return clip

        if (
            n in skip_framelist
        ):
            return clip[1::]
        return clip
    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )        