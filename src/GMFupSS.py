import itertools
import numpy as np
import vapoursynth as vs
import functools
from .GMFupSS_arch import Model_inference
import torch 

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class GMFupSS():
    def __init__(self):
        core = vs.core

        device = torch.device("cuda")
        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model = Model_inference()
        self.model.eval()

    def execute(self, I0, I1):
        with torch.inference_mode():
            middle = self.model(I0, I1).cpu()
        return middle