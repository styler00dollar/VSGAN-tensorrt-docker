from torch import nn as nn
from torch.nn import functional as F
# https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out


# Code mainly from https://github.com/HolyWu/vs-realesrgan
import vapoursynth as vs
import onnx as ox
import onnx_tensorrt.backend as backend
import os
import numpy as np
import torch
from src.SRVGGNetCompact import SRVGGNetCompact
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def SRVGGNetCompactRealESRGAN(clip: vs.VideoNode, scale: int = 2, fp16: bool = False) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if scale not in [2, 4]:
        raise vs.Error('RealESRGAN: scale must be 2 or 4')
    
    # load network
    model_path = f'/workspace/RealESRGANv2-animevideo-xsx{scale}.pth'
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale, act_type='prelu')
    model.load_state_dict(torch.load(model_path, map_location="cpu")['params'])
    model.eval()
    # export to onnx and load with tensorrt (you cant use https://github.com/NVIDIA/Torch-TensorRT because the scripting step will fail)
    torch.onnx.export(model, (torch.rand(1,3,clip.height,clip.width)), f"/workspace/test.onnx", verbose=False, opset_version=13)
    model = ox.load("/workspace/test.onnx")
    model = backend.prepare(model, device='CUDA:0', fp16_mode=fp16)

    def realesrgan(n, f):
        img = frame_to_tensor(f[0])
        output = model.run(img)[0]
        return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_tensor(f):
    arr = np.stack([np.asarray(f.get_read_array(plane) if vs_api_below4 else f[plane]) for plane in range(f.format.num_planes)])
    arr = np.expand_dims(arr, 0)
    return arr


def tensor_to_frame(t, f):
    arr = np.squeeze(t, 0)
    arr = np.clip(arr, 0, 1)
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f.get_write_array(plane) if vs_api_below4 else f[plane]), arr[plane, :, :])
    return f