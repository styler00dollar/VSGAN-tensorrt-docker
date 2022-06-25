# currently only ESRGAN
# current config values for 8gb vram with manjaro, adjust if needed
from src.esrgan import ESRGAN, RealESRGANer
from torchvision.utils import save_image
import torch_tensorrt
import glob
import torch
import os
import cv2  # pip install missing
from tqdm import tqdm

# params
input_folder = "/workspace/tensorrt/input"
output_folder = "/workspace/tensorrt/output"
model_path = "/workspace/4x_fatal_Anime_500000_G.pth"
fp16 = False
tile_x = 670
tile_y = 670
tile_pad = 10
pre_pad = 0

# create output folder if does not exist
if os.path.exists(output_folder) == False:
    os.mkdir(output_folder)

# load files and model
files = glob.glob(input_folder + "/**/*.png", recursive=True)
files_jpg = glob.glob(input_folder + "/**/*.jpg", recursive=True)
files.extend(files_jpg)

model = ESRGAN(model_path)
model.eval()
scale = model.scale

# load model with tensorrt
if fp16 == False:
    model.eval()
    example_data = torch.rand(1, 3, 64, 64)
    model = torch.jit.trace(model, [example_data])
    model = torch_tensorrt.compile(
        model,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 24, 24),
                opt_shape=(1, 3, 500, 500),
                max_shape=(1, 3, 720, 720),
                dtype=torch.float32,
            )
        ],
        enabled_precisions={torch.float},
        truncate_long_and_double=True,
    )
elif fp16 == True:
    # for fp16, the data needs to be on cuda
    model.eval().half().cuda()
    example_data = torch.rand(1, 3, 64, 64).half().cuda()
    model = torch.jit.trace(model, [example_data])
    model = torch_tensorrt.compile(
        model,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 24, 24),
                opt_shape=(1, 3, 500, 500),
                max_shape=(1, 3, 720, 720),
                dtype=torch.half,
            )
        ],
        enabled_precisions={torch.half},
        truncate_long_and_double=True,
    )
    model.half()
del example_data

"""
# you can save the compiled model, and load that instead to skip compile waiting times
# comment the above lines related to model after you saved the model
torch.jit.save(model, "compiled_model.ts")
model = torch.jit.load("compiled_model.ts")
scale = 4
"""

upsampler = RealESRGANer(
    "cuda", scale, model_path, model, tile_x, tile_y, tile_pad, pre_pad
)

for f in tqdm(files):
    image = cv2.imread(f)
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2) / 255
    output = upsampler.enhance(image)
    # save image with opencv
    output = output.cpu().numpy().squeeze(0).swapaxes(0, 2).swapaxes(0, 1) * 255
    cv2.imwrite(
        os.path.join(output_folder, os.path.splitext(os.path.basename(f))[0] + ".png"),
        output,
    )
    # save image with torchvision
    # save_image(output, os.path.join(output_folder, os.path.splitext(os.path.basename(f))[0] + ".png"))
