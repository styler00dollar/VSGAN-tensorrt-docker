import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch
from .stmfnet_arch import STMFNet_Model
from .download import check_and_download

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class STMFNet:
    def __init__(self):
        # clip: vs.VideoNode,
        self.fp16 = False
        self.cache = False
        self.amount_input_img = 4

        core = vs.core
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        model_path = "/workspace/tensorrt/models/stmfnet.pth"
        check_and_download(model_path)
        self.model = STMFNet_Model()
        self.model.load_state_dict(
            torch.load(model_path), True
        )

        self.model.eval().cuda()

        if self.fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.model.half()

    def execute(self, I0, I1, I2, I3):
        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.inference_mode():
            middle = self.model(
                I0,
                I1,
                I2,
                I3,
            )

        middle = middle.detach().squeeze(0).cpu().numpy()
        return middle
