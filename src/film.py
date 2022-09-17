import vapoursynth as vs
import torch
import numpy as np
import kornia
import os
from torch.nn import functional as F
import kornia
import functools

core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

from contextlib import contextmanager
import os
import sys

# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def FILM_inference(clip: vs.VideoNode, model_choise: str = "vgg") -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("This is not a clip")

    if clip.format.id != vs.RGBS:
        raise vs.Error("Only RGBS format is supported")

    if clip.num_frames < 2:
        raise vs.Error("Number of frames must be at least 2")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    with stdout_redirected(to=os.devnull):
        import sys

        if not sys.argv:
            sys.argv.append("(C++)")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        if model_choise == "style":
            model = tf.compat.v2.saved_model.load("/workspace/tensorrt/models/FILM/Style/")
        elif model_choise == "l1":
            model = tf.compat.v2.saved_model.load("/workspace/tensorrt/models/FILM/L1/")
        elif model_choise == "vgg":
            model = tf.compat.v2.saved_model.load("/workspace/tensorrt/models/FILM/VGG/")

    batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    batch_dt = np.expand_dims(batch_dt, axis=0)
    batch_dt = tf.convert_to_tensor(batch_dt)

    w = clip.width
    h = clip.height

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        if (n % 2 == 0) or n == 0:
            return clip

        # if frame number odd
        I0 = frame_to_tensor(clip.get_frame(n - 1))
        I1 = frame_to_tensor(clip.get_frame(n + 1))

        I0 = np.expand_dims(I0, 0)
        I1 = np.expand_dims(I1, 0)

        I0 = np.swapaxes(I0, 3, 1)
        I0 = np.swapaxes(I0, 1, 2)
        I1 = np.swapaxes(I1, 3, 1)
        I1 = np.swapaxes(I1, 1, 2)

        I0 = tf.convert_to_tensor(I0)
        I1 = tf.convert_to_tensor(I1)

        inputs = {"x0": I0, "x1": I1, "time": batch_dt}
        middle = model(inputs, training=False)["image"].numpy()

        middle = np.squeeze(middle, 0)
        middle = np.swapaxes(middle, 0, 2)
        middle = np.swapaxes(middle, 1, 2)
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
