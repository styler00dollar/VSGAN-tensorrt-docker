import vapoursynth as vs
import torch
import numpy as np
import kornia
import os
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


class FILM:
    def __init__(self, model_choise):
        import tensorflow as tf

        with stdout_redirected(to=os.devnull):
            import sys

            if not sys.argv:
                sys.argv.append("(C++)")
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf

            gpus = tf.config.experimental.list_physical_devices("GPU")
            print(gpus)
            tf.config.experimental.set_memory_growth(gpus[0], True)
            if model_choise == "style":
                self.model = tf.compat.v2.saved_model.load(
                    "/workspace/tensorrt/models/FILM/Style/"
                )
            elif model_choise == "l1":
                self.model = tf.compat.v2.saved_model.load(
                    "/workspace/tensorrt/models/FILM/L1/"
                )
            elif model_choise == "vgg":
                self.model = tf.compat.v2.saved_model.load(
                    "/workspace/tensorrt/models/FILM/VGG/"
                )

        self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        self.batch_dt = np.expand_dims(self.batch_dt, axis=0)
        self.batch_dt = tf.convert_to_tensor(self.batch_dt)

    def execute(self, I0, I1):
        I0 = I0.cpu().detach().numpy()
        I1 = I1.cpu().detach().numpy()

        I0 = np.swapaxes(I0, 3, 1)
        I0 = np.swapaxes(I0, 1, 2)
        I1 = np.swapaxes(I1, 3, 1)
        I1 = np.swapaxes(I1, 1, 2)

        I0 = tf.convert_to_tensor(I0)
        I1 = tf.convert_to_tensor(I1)

        inputs = {"x0": I0, "x1": I1, "time": self.batch_dt}
        middle = self.model(inputs, training=False)["image"].numpy()

        middle = np.squeeze(middle, 0)
        middle = np.swapaxes(middle, 0, 2)
        middle = np.swapaxes(middle, 1, 2)
        return middle
