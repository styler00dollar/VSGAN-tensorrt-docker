import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch
from .M2M_arch import M2M_PWC

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class M2M:
    def __init__(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model = M2M_PWC()
        self.model.load_state_dict(torch.load("/workspace/tensorrt/models/M2M.pth"))
        self.model.eval().cuda()
        self.cache = True

    def execute(self, I0, I1, multi):
        tenSteps = [
            torch.FloatTensor([st / (multi) * 1]).view(1, 1, 1, 1).cuda()
            for st in range(1, (multi))
        ]

        with torch.inference_mode():
            output = self.model(I0, I1, tenSteps, multi)
            output = torch.cat(output)

            with open('shape.txt', 'w') as f:
                f.write(str(output.shape))

            with open('tenSteps.txt', 'w') as f:
                f.write(str(tenSteps))

        return output
