from torch import nn as nn
from torch.nn import functional as F
import functools
import numpy as np
import vapoursynth as vs
import os
import numpy as np
import torch

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
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def SRVGGNetCompactRealESRGAN(clip: vs.VideoNode, scale: int = 2, fp16: bool = False, backend: str = "cuda") -> vs.VideoNode:
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
    if backend == "tensorrt":
      import onnx as ox
      import onnx_tensorrt.backend as backend
      # export to onnx and load with tensorrt (you cant use https://github.com/NVIDIA/Torch-TensorRT because the scripting step will fail)
      torch.onnx.export(model, (torch.rand(1,3,clip.height,clip.width)), f"/workspace/test.onnx", verbose=False, opset_version=14)
      model = ox.load("/workspace/test.onnx")
      model = backend.prepare(model, device='CUDA:0', fp16_mode=fp16)
    elif backend == "onnx":
      import onnx as ox
      import onnxruntime as ort
      torch.onnx.export(model, (torch.rand(1,3,clip.height,clip.width)), f"/workspace/test.onnx", verbose=False, opset_version=14, input_names=['input'], output_names=['output'])
      model = ox.load("/workspace/test.onnx")
      sess = ort.InferenceSession(f"/workspace/test.onnx", providers=["CUDAExecutionProvider"])
    elif backend == "quantized_onnx":
      import onnxruntime as ort
      import onnx as ox
      from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_qat, QuantType, quantize, QuantizationMode
      torch.onnx.export(model.cuda(), (torch.rand(1,3,clip.height,clip.width)).cuda(), f"/workspace/test.onnx", verbose=False, opset_version=14, input_names=['input'], output_names=['output'])
      quantized_model = quantize_dynamic("/workspace/test.onnx", "/workspace/test_quant.onnx", weight_type=QuantType.QUInt8)
      model = ox.load("/workspace/test_quant.onnx")
      sess = ort.InferenceSession(f"/workspace/test.onnx", providers=["CUDAExecutionProvider"])
    elif backend == "cuda":
      if fp16:
        model = model.half()
      model.cuda()

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        img = frame_to_tensor(clip.get_frame(n))
        img = np.expand_dims(img, 0)
        
        if backend == "cuda":
          img = torch.Tensor(img).to("cuda", non_blocking=True)
          if fp16:
            img = img.half()
          output = model(img)
          output = output.detach().cpu().numpy()
        elif backend == "onnx" or "quantized_onnx":
          output = sess.run(None, {'input': img})[0]
        else:
          output = model.run(img)[0]
        
        output = np.squeeze(output, 0)
        return tensor_to_clip(clip=clip, image=output)

    return core.std.FrameEval(
            core.std.BlankClip(
                clip=clip,
                width=clip.width * scale,
                height=clip.height * scale
            ),
            functools.partial(
                execute,
                clip=clip
            )
    )

def frame_to_tensor(frame: vs.VideoFrame) -> torch.Tensor:
    return np.stack([
        np.asarray(frame[plane])
        for plane in range(frame.format.num_planes)
    ])

def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
    for plane in range(f.format.num_planes):
        d = np.asarray(f[plane])
        np.copyto(d, array[plane, :, :])
    return f

def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
    clip = core.std.BlankClip(
        clip=clip,
        width=image.shape[-1],
        height=image.shape[-2]
    )
    return core.std.ModifyFrame(
        clip=clip,
        clips=clip,
        selector=lambda n, f: tensor_to_frame(f.copy(), image)
    )
