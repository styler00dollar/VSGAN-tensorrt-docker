import vapoursynth as vs
import os
import torch
import sys
from contextlib import contextmanager
from .download import check_and_download


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


# Code mainly from https://github.com/HolyWu/vs-realesrgan
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4


class compact_inference:
    def __init__(self, scale=2, fp16=True, clip=None):
        self.cache = False
        self.scale = scale
        with stdout_redirected(to=os.devnull):
            # load network
            from .SRVGGNetCompact_arch import SRVGGNetCompact

            model_path = (
                f"/workspace/tensorrt/models/RealESRGANv2-animevideo-xsx{scale}.pth"
            )
            check_and_download(model_path)
            self.model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,
                upscale=scale,
                act_type="prelu",
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")["params"]
            )
            self.model.eval()

            import onnx as ox
            import onnx_tensorrt.backend as backend

            # export to onnx and load with tensorrt (you cant use https://github.com/NVIDIA/Torch-TensorRT because the scripting step will fail)
            torch.onnx.export(
                self.model,
                (torch.rand(1, 3, clip.height, clip.width)),
                "/workspace/tensorrt/models/test.onnx",
                verbose=False,
                opset_version=14,
                # dynamic_axes=dynamic_axes,
                # input_names=["input"],
                # output_names=["output"],
            )
            self.model = ox.load("/workspace/tensorrt/models/test.onnx")
            self.model = backend.prepare(self.model, device="CUDA:0", fp16_mode=fp16)

    def execute(self, input) -> vs.VideoNode:
        output = torch.Tensor(self.model.run(input.cpu().numpy())[0])
        return output
