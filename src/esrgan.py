# https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/constants.py

from typing import OrderedDict

import numpy as np
from torch import Tensor

MAX_DTYPE_VALUES = {
    np.dtype("int8"): 127,
    np.dtype("uint8"): 255,
    np.dtype("int16"): 32767,
    np.dtype("uint16"): 65535,
    np.dtype("int32"): 2147483647,
    np.dtype("uint32"): 4294967295,
    np.dtype("int64"): 9223372036854775807,
    np.dtype("uint64"): 18446744073709551615,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
}
STATE_T = OrderedDict[str, Tensor]


# https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/archs/blocks.py
"""
Common Blocks between Architectures.
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type: Optional[str] = 'relu', mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """
    if mode not in ['CNA', 'NAC', 'CNAC']:
        raise AssertionError('Wong conv mode [%s]' % mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias,
                  groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    if mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Upsampler
####################


def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                       pad_type='zero', norm_type=None, act_type='relu'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor ** 2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True,
                 pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias,
                      pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)



# https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/archs/ESRGAN.py
import math
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch import pixel_unshuffle

#from vsgan.archs import blocks as block
#from vsgan.constants import STATE_T


class ESRGAN(nn.Module):
    def __init__(self, model: str, norm=None, act: str = "leakyrelu", upsampler: str = "upconv",
                 mode: str = "CNA") -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.

        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.

        This network supports model files from both new and old-arch.

        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super(ESRGAN, self).__init__()

        self.model = model
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # wanted, possible key names
            # currently supports old, new, and newer RRDBNet arch models
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            "model.3.weight": ("upconv1.weight", "conv_up1.weight"),
            "model.3.bias": ("upconv1.bias", "conv_up1.bias"),
            "model.6.weight": ("upconv2.weight", "conv_up2.weight"),
            "model.6.bias": ("upconv2.bias", "conv_up2.bias"),
            "model.8.weight": ("HRconv.weight", "conv_hr.weight"),
            "model.8.bias": ("HRconv.bias", "conv_hr.bias"),
            "model.10.weight": ("conv_last.weight",),
            "model.10.bias": ("conv_last.bias",),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)"
            )
        }
        self.state = torch.load(self.model)
        if "params_ema" in self.state:
            self.state = self.state["params_ema"]
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())

        self.state: STATE_T = self.new_to_old_arch(self.state)
        self.in_nc = self.state["model.0.weight"].shape[1]
        self.out_nc = self.get_out_nc() or self.in_nc  # assume same as in nc if not found
        self.scale = self.get_scale()
        self.num_filters = self.state["model.0.weight"].shape[0]

        if self.in_nc in (self.out_nc * 4, self.out_nc * 16) and \
                self.out_nc in (self.in_nc / 4, self.in_nc / 16):
            self.shuffle_factor = int(math.sqrt(self.in_nc / self.out_nc))
        else:
            self.shuffle_factor = None

        upsample_block = {
            "upconv": upconv_block,
            "pixel_shuffle": pixelshuffle_block
        }.get(self.upsampler)
        if upsample_block is None:
            raise NotImplementedError("Upsample mode [%s] is not found" % self.upsampler)

        if self.scale == 3:
            upsample_blocks = upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                upscale_factor=3,
                act_type=self.act
            )
        else:
            upsample_blocks = [upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                act_type=self.act
            ) for _ in range(int(math.log(self.scale, 2)))]

        self.model = sequential(
            # fea conv
            conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None
            ),
            ShortcutBlock(sequential(
                # rrdb blocks
                *[RRDB(
                    nc=self.num_filters,
                    kernel_size=3,
                    gc=32,
                    stride=1,
                    bias=True,
                    pad_type="zero",
                    norm_type=self.norm,
                    act_type=self.act,
                    mode="CNA",
                    plus=self.plus
                ) for _ in range(self.num_blocks)],
                # lr conv
                conv_block(
                    in_nc=self.num_filters,
                    out_nc=self.num_filters,
                    kernel_size=3,
                    norm_type=self.norm,
                    act_type=None,
                    mode=self.mode
                )
            )),
            *upsample_blocks,
            # hr_conv0
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act
            ),
            # hr_conv1
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None
            )
        )

        # vapoursynth calls expect the real scale even if shuffled
        if self.shuffle_factor:
            self.scale = self.scale // self.shuffle_factor

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state: STATE_T) -> STATE_T:
        """Convert a new-arch model state dictionary to an old-arch dictionary."""
        if "params_ema" in state:
            state = state["params_ema"]

        if "conv_first.weight" not in state:
            # model is already old arch, this is a loose check, but should be sufficient
            return state

        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[f"model.1.sub./NB/.{kind}"]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        return old_state

    def get_out_nc(self) -> Optional[int]:
        max_part = 0
        out_nc = None
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > max_part:
                    max_part = part_num
                    out_nc = self.state[part].shape[0]
        return out_nc

    def get_scale(self, min_part: int = 6) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2 ** n

    def get_num_blocks(self) -> int:
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x):
        if self.shuffle_factor:
            x = pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
        return self.model(x)


class ResidualDenseBlock5C(nn.Module):
    """
    5 Convolution Residual Dense 
    Residual Dense Network for Image Super-Resolution, CVPR 18.
    gc: growth channel, i.e. intermediate channels
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type="zero", norm_type=None,
                 act_type="leakyrelu", mode="CNA", plus=False):
        super(ResidualDenseBlock5C, self).__init__()
        last_act = None if mode == "CNA" else act_type

        self.conv1x1 = conv1x1(nc, gc) if plus else None
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type,
                                      norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """Residual in Residual Dense """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type="zero", norm_type=None,
                 act_type="leakyrelu", mode="CNA", plus=False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus)
        self.RDB2 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus)
        self.RDB3 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode, plus)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out.mul(0.2) + x


def conv1x1(in_planes: int, out_planes: int, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# https://github.com/HolyWu/vs-realesrgan/blob/master/vsrealesrgan/__init__.py
import os
import functools
import numpy as np
import torch
import vapoursynth as vs
from .realesrganner import RealESRGANer
core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4


def ESRGAN_inference(clip: vs.VideoNode, model_path: str = "/workspace/4x_fatal_Anime_500000_G.pth", tile_x: int = 0, tile_y: int = 0, tile_pad: int = 10, pre_pad: int = 0,
               device_type: str = 'cuda', device_index: int = 0, fp16: bool = False) -> vs.VideoNode:

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("RealESRGAN: device_type must be 'cuda' or 'cpu'")

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model = ESRGAN(model_path)
    model.eval()
    scale = model.scale

    import torch_tensorrt
    if fp16 == False:
        model.eval()
        example_data = torch.rand(1,3,64,64)
        model = torch.jit.trace(model, [example_data])
        model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input( \
                        min_shape=(1, 3, 24, 24), \
                        opt_shape=(1, 3, 256, 256), \
                        max_shape=(1, 3, 512, 512), \
                        dtype=torch.float32)], \
                        enabled_precisions={torch.float}, truncate_long_and_double=True)
    elif fp16 == True:
        # for fp16, the data needs to be on cuda
        model.eval().half().cuda()
        example_data = torch.rand(1,3,64,64).half().cuda()
        model = torch.jit.trace(model, [example_data])
        model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input( \
                        min_shape=(1, 3, 24, 24), \
                        opt_shape=(1, 3, 256, 256), \
                        max_shape=(1, 3, 512, 512), \
                        dtype=torch.half)], \
                        enabled_precisions={torch.half}, truncate_long_and_double=True)
        model.half()
    del example_data

    upsampler = RealESRGANer(device, scale, model_path, model, tile_x, tile_y, tile_pad, pre_pad)

    @torch.inference_mode()
    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        img = frame_to_tensor(clip.get_frame(n))
        img = torch.Tensor(img).unsqueeze(0).to(device, non_blocking=True)

        if fp16 == True:
            img = img.half()
        
        output = upsampler.enhance(img)
        output = output.detach().squeeze().cpu().numpy()

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