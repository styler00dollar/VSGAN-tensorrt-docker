import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class IFRNet() :
    def __init__(self, model, fp16):
        self.fp16 = fp16

        if model == "small":
            from .IFRNet_S_arch import IRFNet_S

            self.model = IRFNet_S()
            self.model.load_state_dict(torch.load("/workspace/tensorrt/models/IFRNet_S_Vimeo90K.pth"))

        elif model == "large":
            from .IFRNet_L_arch import IRFNet_L

            self.model = IRFNet_L()
            self.model.load_state_dict(torch.load("/workspace/tensorrt/models/IFRNet_L_Vimeo90K.pth"))
        self.model.eval().cuda()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def execute(self, I0, I1):
        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.inference_mode():
            middle = self.model(I0, I1)

        return middle.detach().squeeze(0).cpu().numpy()