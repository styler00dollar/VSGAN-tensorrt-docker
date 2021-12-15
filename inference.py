"""
15-Dez-21
https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/RRDBNet.py
https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/RRDBNet_Blocks.py
https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/__init__.py
https://github.com/rlaphoenix/VSGAN/blob/master/vsgan/constants.py
"""
from __future__ import annotations
from collections import OrderedDict
from torch.nn import functional as F
from torch import nn as nn
from typing import Optional
from typing import Union
from vapoursynth import core
import functools
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import vapoursynth as vs

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
# Useful blocks
####################


class ResNetBlock(nn.Module):
    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type,
                           mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, norm_type, act_type,
                           mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock5C(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None,
                 act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                                act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc + gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                                act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc + 2 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                                act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc + 3 * gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                                act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc + 4 * gc, nc, 3, stride, bias=bias, pad_type=pad_type, norm_type=norm_type,
                                act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', norm_type=None,
                 act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock5C(nc, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


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


class RRDBNet(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int, upscale: int = 4, norm_type=None,
                 act_type: str = 'leakyrelu', mode: str = 'CNA', upsample_mode='upconv'):
        """
        Residual in Residual Dense Block Network.

        This is specifically v0.1 (aka old-arch) and is not the newest revision code
        that's available at github:/xinntao/ESRGAN. This is on purpose, the newest
        code has hardcoded and severely limited the potential use of the Network.
        Specifically it has hardcoded the scale value to be `4` no matter what.

        :param in_nc: Input number of channels
        :param out_nc: Output number of channels
        :param nf: Number of filters
        :param nb: Number of blocks
        :param upscale: Scale relative to input
        :param norm_type: Normalization type
        :param act_type: Activation type
        :param mode: Convolution mode
        :param upsample_mode: Upsample block type. upconv, pixel_shuffle
        """
        super(RRDBNet, self).__init__()

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = conv_block(in_nc=in_nc, out_nc=nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [RRDB(
            nc=nf,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type='zero',
            norm_type=norm_type,
            act_type=act_type,
            mode='CNA'
        ) for _ in range(nb)]
        lr_conv = conv_block(in_nc=nf, out_nc=nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = upconv_block
        elif upsample_mode == 'pixel_shuffle':
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(in_nc=nf, out_nc=nf, upscale_factor=3, act_type=act_type)
        else:
            upsampler = [upsample_block(in_nc=nf, out_nc=nf, act_type=act_type) for _ in range(n_upscale)]

        hr_conv0 = conv_block(in_nc=nf, out_nc=nf, kernel_size=3, norm_type=None, act_type=act_type)
        hr_conv1 = conv_block(in_nc=nf, out_nc=out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, lr_conv)),
            *upsampler,
            hr_conv0,
            hr_conv1
        )

    def forward(self, x):
        return self.model(x)



class VSGAN:
    def __init__(self, clip: vs.VideoNode, device: Union[str, int] = "cuda"):
        """
        Create a PyTorch Device instance, to use VSGAN with.
        It validates the supplied pytorch device identifier, and makes sure CUDA environment is available and ready.
        :param device: PyTorch device identifier, tells VSGAN which device to run ESRGAN with. e.g. `cuda`, `0`, `1`
        """
        device = device.strip().lower() if isinstance(device, str) else device
        if device == "":
            raise ValueError("VSGAN: `device` parameter cannot be an empty string.")
        if device == "cpu":
            raise ValueError(
                "VSGAN: Using your CPU as a device for VSGAN/PyTorch has been blocked, use a GPU device.\n"
                "Using ESRGAN on a CPU will run it at very high utilisation and temps and may straight up kill it.\n"
                "It isn't worth it either as it takes literally hours for a single 720x480 frame.\n"
                "If you are sure you would like to use your CPU, then use `cpu!` as the device argument."
            )
        if device == "cpu!":
            device = "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise EnvironmentError("VSGAN: Either NVIDIA CUDA or the device (%s) isn't available." % device)
        self.device = device
        self.torch_device = torch.device(self.device)
        self.clip = clip
        self.model = None
        self.model_scale = None
        self.rrdb_net_model = None

    def load_model(self, model: str) -> VSGAN:
        """
        Load an ESRGAN model file into the VSGAN object instance.
        The model can be changed by calling load_model at any point.
        :param model: ESRGAN .pth model file.
        """
        self.model = model
        state_dict = self.sanitize_state_dict(torch.load(self.model))
        # extract model information
        scale2 = 0
        max_part = 0
        scale_min = 6
        nb = None
        out_nc = None
        for part in list(state_dict):
            parts = part.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > scale_min and parts[0] == "model" and parts[2] == "weight":
                    scale2 += 1
                if part_num > max_part:
                    max_part = part_num
                    out_nc = state_dict[part].shape[0]
        self.model_scale = 2 ** scale2
        in_nc = state_dict["model.0.weight"].shape[1]
        nf = state_dict["model.0.weight"].shape[0]

        if nb is None:
            raise NotImplementedError("VSGAN: Could not find the nb in this new-arch model.")
        if out_nc is None:
            print("VSGAN Warning: Could not find out_nc, assuming it's the same as in_nc...")

        self.rrdb_net_model = RRDBNet(in_nc, out_nc or in_nc, nf, nb, self.model_scale)
        self.rrdb_net_model.load_state_dict(state_dict, strict=False)
        self.rrdb_net_model.eval()
        self.rrdb_net_model = self.rrdb_net_model.to(self.torch_device)
        """
        import torch_tensorrt
        example_data = torch.rand(1,3,64,64)
        self.rrdb_net_model = torch.jit.trace(self.rrdb_net_model, [example_data])
        self.rrdb_net_model = torch_tensorrt.compile(self.rrdb_net_model, inputs=[torch_tensorrt.Input( \
                        min_shape=(1, 3, 64, 64), \
                        opt_shape=(1, 3, 256, 256), \
                        max_shape=(1, 3, 512, 512), \
                        dtype=torch.float32)], \
                        enabled_precisions={torch.float}, truncate_long_and_double=True)
        del example_data
        """

        return self

    def run(self, overlap: int = 0) -> VSGAN:
        """
        Executes VSGAN on the provided clip, returning the resulting in a new clip.
        :param overlap: Reduces VRAM usage by seamlessly rendering the input frame(s) in quadrants.
            This reduces memory usage but may also reduce speed. Only use this to stretch your VRAM.
        :returns: ESRGAN result clip
        """
        if self.clip.format.color_family.name != "RGB":
            raise ValueError(
                "VSGAN: Clip color format must be RGB as the ESRGAN model can only work with RGB data :(\n"
                "You can use mvsfunc.ToRGB or use the format option on core.resize functions.\n"
                "The clip might need to be bit depth of 8bpp for correct color input/output.\n"
                "If you need to specify a kernel for chroma, I recommend Spline or Bicubic."
            )

        self.clip = core.std.FrameEval(
            core.std.BlankClip(
                clip=self.clip,
                width=self.clip.width * self.model_scale,
                height=self.clip.height * self.model_scale
            ),
            functools.partial(
                self.execute,
                clip=self.clip,
                overlap=overlap
            )
        )

        #return self
        return self.clip

    def execute(self, n: int, clip: vs.VideoNode, overlap: int = 0) -> vs.VideoNode:
        """
        Run the ESRGAN repo's Modified ESRGAN RRDBNet super-resolution code on a clip's frame.
        Unlike the original code, frames are modified directly as Tensors, without CV2.

        Thanks to VideoHelp for initial support, and @JoeyBallentine for his work on
        seamless chunk support.
        """
        if not self.rrdb_net_model:
            raise ValueError("VSGAN: No ESRGAN model has been loaded, use VSGAN.load_model().")

        def scale(quadrant: torch.Tensor) -> torch.Tensor:
            try:
                quadrant = quadrant.to(self.torch_device)
                with torch.no_grad():
                    return self.rrdb_net_model(quadrant).data
            except RuntimeError as e:
                if "allocate" in str(e) or "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise

        lr_img = self.frame_to_tensor(clip.get_frame(n))

        if not overlap:
            output_img = scale(lr_img)
        elif overlap > 0:
            b, c, h, w = lr_img.shape

            out_h = h * self.model_scale
            out_w = w * self.model_scale
            output_img = torch.empty(
                (b, c, out_h, out_w), dtype=lr_img.dtype, device=lr_img.device
            )

            top_left_sr = scale(lr_img[..., : h // 2 + overlap, : w // 2 + overlap])
            top_right_sr = scale(lr_img[..., : h // 2 + overlap, w // 2 - overlap:])
            bottom_left_sr = scale(lr_img[..., h // 2 - overlap:, : w // 2 + overlap])
            bottom_right_sr = scale(lr_img[..., h // 2 - overlap:, w // 2 - overlap:])

            output_img[..., : out_h // 2, : out_w // 2] = top_left_sr[..., : out_h // 2, : out_w // 2]
            output_img[..., : out_h // 2, -out_w // 2:] = top_right_sr[..., : out_h // 2, -out_w // 2:]
            output_img[..., -out_h // 2:, : out_w // 2] = bottom_left_sr[..., -out_h // 2:, : out_w // 2]
            output_img[..., -out_h // 2:, -out_w // 2:] = bottom_right_sr[..., -out_h // 2:, -out_w // 2:]
        else:
            raise ValueError("Invalid overlap. Must be a value greater than 0, or a False-y value to disable.")

        return self.tensor_to_clip(clip, output_img)

    @staticmethod
    def sanitize_state_dict(state_dict: dict) -> dict:
        """
        Convert a new-arch model state dictionary to an old-arch dictionary.
        The new-arch model's only purpose is making the dict keys more verbose, but has no purpose other
        than that. So to easily support both new and old arch models, simply convert the key names back
        to their "Old" counterparts.

        :param state_dict: new-arch state dictionary
        :returns: old-arch state dictionary
        """
        if "conv_first.weight" not in state_dict:
            # model is already old arch, this is a loose check, but should be sufficient
            return state_dict
        old_net = {
            "model.0.weight": state_dict["conv_first.weight"],
            "model.0.bias": state_dict["conv_first.bias"],
            "model.1.sub.23.weight": state_dict["trunk_conv.weight"],
            "model.1.sub.23.bias": state_dict["trunk_conv.bias"],
            "model.3.weight": state_dict["upconv1.weight"],
            "model.3.bias": state_dict["upconv1.bias"],
            "model.6.weight": state_dict["upconv2.weight"],
            "model.6.bias": state_dict["upconv2.bias"],
            "model.8.weight": state_dict["HRconv.weight"],
            "model.8.bias": state_dict["HRconv.bias"],
            "model.10.weight": state_dict["conv_last.weight"],
            "model.10.bias": state_dict["conv_last.bias"]
        }
        for key, value in state_dict.items():
            if "RDB" in key:
                new = key.replace("RRDB_trunk.", "model.1.sub.")
                if ".weight" in key:
                    new = new.replace(".weight", ".0.weight")
                elif ".bias" in key:
                    new = new.replace(".bias", ".0.bias")
                old_net[new] = value
        return old_net

    @staticmethod
    def frame_to_np(frame: vs.VideoFrame) -> np.dstack:
        """
        Alternative to cv2.imread() that will directly read images to a numpy array.
        :param frame: VapourSynth frame from a clip
        """
        return np.dstack([np.asarray(frame[i]) for i in range(frame.format.num_planes)])

    @staticmethod
    def frame_to_tensor(frame: vs.VideoFrame, change_range=True, bgr2rgb=False, add_batch=True, normalize=False) \
            -> torch.Tensor:
        """
        Read an image as a numpy array and convert it to a tensor.
        :param frame: VapourSynth frame from a clip.
        :param normalize: Normalize (z-norm) from [0,1] range to [-1,1].
        """
        array = VSGAN.frame_to_np(frame)

        if change_range:
            max_val = MAX_DTYPE_VALUES.get(array.dtype, 1.0)
            array = array.astype(np.dtype("float32")) / max_val

        array = torch.from_numpy(
            np.ascontiguousarray(np.transpose(array, (2, 0, 1)))  # HWC->CHW
        ).float()

        if bgr2rgb:
            if array.shape[0] % 3 == 0:
                # RGB or MultixRGB (3xRGB, 5xRGB, etc. For video tensors.)
                array = array.flip(-3)
            elif array.shape[0] == 4:
                # RGBA
                array = array[[2, 1, 0, 3], :, :]

        if add_batch:
            # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
            array.unsqueeze_(0)

        if normalize:
            array = ((array - 0.5) * 2.0).clamp(-1, 1)

        return array

    @staticmethod
    def tensor_to_frame(f: vs.VideoFrame, t: torch.Tensor) -> vs.VideoFrame:
        """
        Copies each channel from a Tensor into a vs.VideoFrame.
        It expects the tensor array to have the dimension count (C) first in the shape, so CHW or CWH.
        :param f: VapourSynth frame to store retrieved planes.
        :param t: Tensor array to retrieve planes from.
        :returns: New frame with planes from tensor array
        """
        array = t.squeeze(0).detach().cpu().clamp(0, 1).numpy()

        d_type = np.asarray(f[0]).dtype
        array = MAX_DTYPE_VALUES.get(d_type, 1.0) * array
        array = array.astype(d_type)

        for plane in range(f.format.num_planes):
            d = np.asarray(f[plane])
            np.copyto(d, array[plane, :, :])
        return f

    def tensor_to_clip(self, clip: vs.VideoNode, image: torch.Tensor) -> vs.VideoNode:
        """
        Convert a tensor into a VapourSynth clip.
        :param clip: used to inherit expected return properties only
        :param image: tensor (expecting CHW shape order)
        :returns: VapourSynth clip with the frame applied
        """
        batch, planes, height, width = image.size()
        clip = core.std.BlankClip(
            clip=clip,
            width=width,
            height=height
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: self.tensor_to_frame(f.copy(), image)
        )


core = vs.core
vs_api_below4 = vs.__api_version__.api_major < 4
core = vs.core
core.num_threads = 16
core.std.LoadPlugin(path='/usr/lib/x86_64-linux-gnu/libffms2.so')
clip = core.ffms2.Source(source='input.webm')
# convert colorspace
#clip = vs.core.resize.Bicubic(clip, format=vs.RGBS, matrix_in_s='709')
# convert colorspace + resizing
clip = vs.core.resize.Bicubic(clip, width=848, height=480, format=vs.RGBS, matrix_in_s='709')
# currently only taking normal esrgan models
clip = VSGAN(clip, device="cuda").load_model("model.pth").run(overlap=16)
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
clip.set_output()