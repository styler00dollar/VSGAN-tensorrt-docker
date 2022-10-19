# https://github.com/HolyWu/vs-realesrgan/blob/5a1bef43acb18c346457df717238ac3cebc8c656/vsrealesrgan/utils.py#L34
import math
import torch
from torch.nn import functional as F


class RealESRGANer:
    def __init__(
        self,
        scale,
        model=None,
        tile_x=0,
        tile_y=0,
        tile_pad=10,
        pre_pad=10,
    ):
        self.scale = scale
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.model = model

    def pre_process(self, img):
        self.img = img

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        # mod pad
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if h % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - w % self.mod_scale
            self.img = F.pad(
                self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect"
            )

    def process(self):
        self.output = self.model.execute(self.img)

    def tile_process(self):
        """Modified from: https://github.com/ata4/esrgan-launcher"""
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_x)
        tiles_y = math.ceil(height / self.tile_y)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_x
                ofs_y = y * self.tile_y

                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_x, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_y, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                output_tile = self.model.execute(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.mod_pad_h * self.scale,
                0 : w - self.mod_pad_w * self.scale,
            ]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : h - self.pre_pad * self.scale,
                0 : w - self.pre_pad * self.scale,
            ]
        return self.output

    def enhance(self, img):
        self.pre_process(img)
        if self.tile_x > 0 and self.tile_y > 0:
            self.tile_process()
        else:
            self.process()
        return self.post_process()
