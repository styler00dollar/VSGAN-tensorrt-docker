# https://github.com/AmusementClub/vs-mlrt/blob/7856cd3aab7111345475f7d526fe75ef81e198c8/scripts/vsmlrt.py
# https://github.com/mafiosnik777/enhancr/blob/cf763fee358e8a2483154ad0d59a9607f4f4bdba/src/env/inference/utils/trt_precision.py

import vapoursynth as vs
from vapoursynth import core
import tensorrt as trt
from polygraphy.backend.trt import TrtRunner
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_model_variables(model_path):
    with open(model_path, "rb") as f:
        engine_data = f.read()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    runner = TrtRunner(engine)
    with runner:
        input_metadata = runner.get_input_metadata()
        input_precision = input_metadata["input"].dtype
        num_channels_in = input_metadata["input"].shape[1]
        return input_precision, num_channels_in


def rife_trt(
    clip: vs.VideoNode,
    multi: int = 2,
    scale: float = 1.0,
    device_id: int = 0,
    num_streams: int = 1,
    engine_path: str = "",
):

    model_precision, model_in_channels = get_model_variables(engine_path)
    if model_precision == "float32":
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

    if model_in_channels == 8:
        scale = core.std.Interleave(
            [
                clip.std.BlankClip(format=grayPrecision, color=scale, length=1)
                for i in range(1, multi)
            ]
        ).std.Loop(clip.num_frames)

        clips = [initial, terminal, timepoint, scale]
    elif model_in_channels == 7:
        eprint("Scale parameter will be ignored in v2 models")
        clips = [initial, terminal, timepoint]
    else:
        raise ValueError("Invalid model, channel mismatch")

    output = core.trt.Model(
        clips, engine_path, device_id=device_id, num_streams=num_streams
    )

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
