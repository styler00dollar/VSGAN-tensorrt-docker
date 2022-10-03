import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch
from .M2M_arch import M2M_PWC

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class M2M():
    def __init__(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model = M2M_PWC()
        self.model.load_state_dict(torch.load("/workspace/tensorrt/models/M2M.pth"))

        self.model.eval().cuda()

    def execute(self, I0, I1):
        with torch.inference_mode():
            intRatio = None
            intStep = 0.5
            tenSteps = torch.FloatTensor([0.5]).view(1, 1, 1, 1).cuda()
            middle = self.model(I0, I1, tenSteps, intRatio)[0]

        return middle.detach().squeeze(0).cpu().numpy()