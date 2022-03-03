# https://github.com/bilibili/ailab/blob/main/Real-CUGAN/VapourSynth/upcunet_v3_vs.py
import torch
from torch import nn as nn
from torch.nn import functional as F
import os,sys
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x,x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x
class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z
class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x1,x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z
class UNet1x3(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1x3, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 5, 3, 2)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x1,x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z
class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):#conv234结尾有se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2,x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1,x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

class UpCunet2x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x):#1.7G
        n, c, h0, w0 = x.shape
        #if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')  # 需要保证被2整除
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
        return x
        
class UpCunet3x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet3x, self).__init__()
        self.unet1 = UNet1x3(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x):#1.7G
        n, c, h0, w0 = x.shape
        #if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 4 + 1) * 4
        pw = ((w0 - 1) // 4 + 1) * 4
        x = F.pad(x, (14, 14 + pw - w0, 14, 14 + ph - h0), 'reflect')  # 需要保证被2整除
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 3, :w0 * 3]
        return x

class UpCunet4x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet4x, self).__init__()
        self.unet1 = UNet1(in_channels, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps=nn.PixelShuffle(2)
        self.conv_final=nn.Conv2d(64,12,3,1,padding=0,bias=True)
    def forward(self, x):
        n, c, h0, w0 = x.shape
        x00 = x
        #if(tile_mode==0):#不tile
        
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (19, 19 + pw - w0, 19, 19 + ph - h0), 'reflect')  # 需要保证被2整除
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        x=self.conv_final(x)
        x=F.pad(x,(-1,-1,-1,-1))
        x=self.ps(x)
        if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 4, :w0 * 4]
        x+=F.interpolate(x00, scale_factor=4, mode='nearest')
        return x
        
        
class pixel_unshuffle(nn.Module):
    def __init__(self, ratio=2):
        super(pixel_unshuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)

class UpCunet2x_fast(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x_fast, self).__init__()
        self.unet1 = UNet1(12, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps=nn.PixelShuffle(2)
        self.conv_final=nn.Conv2d(64,12,3,1,padding=0,bias=True)
        self.inv=pixel_unshuffle(2)

    def forward(self, x):
        n, c, h0, w0 = x.shape
        x00 = x
        #if(tile_mode==0):#不tile

        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (38, 38 + pw - w0, 38, 38 + ph - h0), 'reflect')  # 需要保证被2整除
        x = self.inv(x)  # +18
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        x=self.conv_final(x)
        # with open(r"C:\Users\liujing\Desktop\log.txt","a+")as f:
        #     f.write("%s"%(str(x.shape)))
        #     f.flush()
        x=F.pad(x,(-1,-1,-1,-1))
        x=self.ps(x)
        if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
        x+=F.interpolate(x00, scale_factor=2, mode='nearest')
        return x


# Code mainly from https://github.com/HolyWu/vs-realesrgan
import vapoursynth as vs
import os
import functools
from .realesrganner import RealESRGANer
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4

def cugan_inference(clip: vs.VideoNode, fp16: bool = True, scale: int = 2, kind_model: str = "no_denoise", backend_inference: str = "cuda", tile_x:int = 512, tile_y:int = 512, tile_pad:int = 10, pre_pad:int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('cugan: This is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('cugan: only RGBS format is supported')
    
    if scale == 2:
      model = UpCunet2x(in_channels=3, out_channels=3)
      if kind_model == "no_denoise":
        model_path = "/workspace/up2x-latest-no-denoise.pth"
      elif kind_model == "conservative":
        model_path = "/workspace/up2x-latest-conservative.pth"
      elif kind_model == "denoise3x":
        model_path = "/workspace/up2x-latest-denoise3x.pth"
    elif scale == 3:
      model = UpCunet3x(in_channels=3, out_channels=3)
      if kind_model == "no_denoise":
        model_path = "/workspace/up3x-latest-no-denoise.pth"
      elif kind_model == "conservative":
        model_path = "/workspace/up3x-latest-conservative.pth"
      elif kind_model == "denoise3x":
        model_path = "/workspace/up3x-latest-denoise3x.pth"
    elif scale == 4:
      model = UpCunet4x(in_channels=3, out_channels=3)
      if kind_model == "no_denoise":
        model_path = "/workspace/up4x-latest-no-denoise.pth"
      elif kind_model == "conservative":
        model_path = "/workspace/up4x-latest-conservative.pth"
      elif kind_model == "denoise3x":
        model_path = "/workspace/up4x-latest-denoise3x.pth"

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    if backend_inference == "cuda":
      model.eval().cuda()
      if fp16:
        model = model.half()
      upsampler = RealESRGANer("cuda", scale, model_path, model, tile_x, tile_y, tile_pad, pre_pad)

    elif backend_inference == "onnx":
        import onnx as ox
        import onnxruntime as ort
        if fp16:
          torch.onnx.export(model.eval().half().cuda(), torch.rand(1,3,clip.height,clip.width).half().cuda(), "/workspace/cugan.onnx", verbose=False, input_names=["input"], output_names=["output"], opset_version=14)
        else:
          torch.onnx.export(model.eval().cuda(), torch.rand(1,3,clip.height,clip.width).cuda(), "/workspace/cugan.onnx", verbose=False, input_names=["input"], output_names=["output"], opset_version=14)
        model = ox.load("/workspace/cugan.onnx")
        sess = ort.InferenceSession(f"/workspace/cugan.onnx", providers=["CUDAExecutionProvider"])

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        img = frame_to_tensor(clip.get_frame(n))
        
        if backend_inference == "cuda":
          img = torch.Tensor(img).unsqueeze(0).to("cuda", non_blocking=True)
          if fp16:
            img = img.half()
          #output = model(img)
          output = upsampler.enhance(img)
          output = output.detach().squeeze(0).cpu().numpy()
        elif backend_inference == "onnx":
          img = np.expand_dims(img, 0)
          if fp16:
            img = img.astype(np.float16)
          output = sess.run(None, {'input': img})[0]
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

def frame_to_tensor(frame: vs.VideoFrame):
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