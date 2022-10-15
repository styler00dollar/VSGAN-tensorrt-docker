import itertools
import numpy as np
import vapoursynth as vs
import functools
from .eisai_arch import SoftsplatLite, DTM, RAFT, interpolate
import torch

# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class EISAI:
    def __init__(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # load models
        device = torch.device("cuda")

        ssl = SoftsplatLite()
        dtm = DTM()
        ssl.load_state_dict(torch.load("/workspace/tensorrt/models/eisai_ssl.pt"))
        dtm.load_state_dict(torch.load("/workspace/tensorrt/models/eisai_dtm.pt"))
        self.raft = (
            RAFT(path="/workspace/tensorrt/models/eisai_anime_interp_full.ckpt")
            .eval()
            .to(device)
        )
        self.ssl = ssl.to(device).eval()
        self.dtm = dtm.to(device).eval()

    def execute(self, I0, I1, timestep):
        with torch.inference_mode():
            middle = interpolate(self.raft, self.ssl, self.dtm, I0, I1, t=timestep)
            middle = middle.detach().cpu().numpy()
        return middle
