# https://github.com/AmusementClub/vs-mlrt/blob/7856cd3aab7111345475f7d526fe75ef81e198c8/scripts/vsmlrt.py
# https://github.com/mafiosnik777/enhancr/blob/cf763fee358e8a2483154ad0d59a9607f4f4bdba/src/env/inference/utils/trt_precision.py

import vapoursynth as vs
from vapoursynth import core
from .vfi_inference import vfi_frame_merger
import tensorrt as trt
from polygraphy.backend.trt import TrtRunner


def check_model_precision_trt(model_path):
    with open(model_path, "rb") as f:
        engine_data = f.read()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    runner = TrtRunner(engine)
    with runner:
        input_metadata = runner.get_input_metadata()
        input_precision = input_metadata["input"].dtype
        return input_precision


def rife_trt(
    clip: vs.VideoNode,
    multi: int = 2,
    scale: float = 1.0,
    device_id: int = 0,
    num_streams: int = 1,
    engine_path: str = "",
):
    if check_model_precision_trt(engine_path) == "float32":
        grayPrecision = vs.GRAYS
    else:
        grayPrecision = vs.GRAYH

    initial = core.std.Interleave([clip] * (multi - 1))

    terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
    terminal = core.std.Interleave([terminal] * (multi - 1))

    timepoint = core.std.Interleave(
        [
            clip.std.BlankClip(format=grayPrecision, color=i / multi, length=1)
            for i in range(1, multi)
        ]
    ).std.Loop(clip.num_frames)

    scale = core.std.Interleave(
        [
            clip.std.BlankClip(format=grayPrecision, color=scale, length=1)
            for i in range(1, multi)
        ]
    ).std.Loop(clip.num_frames)

    clips = [initial, terminal, timepoint, scale]

    output = core.trt.Model(
        clips, engine_path, device_id=device_id, num_streams=num_streams
    )

    # using FrameEval is much faster than calling core.akarin.Expr
    output = vfi_frame_merger(initial, output)

    if multi == 2:
        return core.std.Interleave([clip, output])
    else:
        return core.std.Interleave(
            [
                clip,
                *(
                    output.std.SelectEvery(cycle=multi - 1, offsets=i)
                    for i in range(multi - 1)
                ),
            ]
        )
