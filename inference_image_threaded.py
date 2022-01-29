# currently only for compact and forces fp16
from src.SRVGGNetCompact import SRVGGNetCompact
import glob
import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from turbojpeg import TurboJPEG
jpeg_reader = TurboJPEG()  

# params
input_folder = "input/"
output_folder = "output/"
pool_size = 4

# create output folder if does not exist
if os.path.exists(output_folder) == False:
    os.mkdir(output_folder)

# load files and model
files = glob.glob(input_folder + '/**/*.png', recursive=True)
files_jpg = glob.glob(input_folder + '/**/*.jpg', recursive=True)
files.extend(files_jpg)
files.sort()

scale = 2
model_path = f'/workspace/RealESRGANv2-animevideo-xsx{scale}.pth'
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=scale, act_type='prelu')
model.load_state_dict(torch.load(model_path, map_location="cpu")['params'])
model.eval().half().cuda()

def worker(f):
    try:
      image = jpeg_reader.decode(open(f, "rb").read(), 0)
      image = torch.from_numpy(image).half().unsqueeze(0).permute(0,3,1,2)/255
      image = image.to("cuda", non_blocking=True)
      out = model(image)

      out = out*255
      out = out.clamp(0,255)
      out = out.detach().cpu().numpy()
      out = out.squeeze(0).swapaxes(0, 2).swapaxes(0, 1).astype(np.uint8)
      out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(output_folder, os.path.splitext(os.path.basename(f))[0] + ".jpg"), out)
    except Exception as e: print(e)

with Pool(pool_size) as p:
  r = list(tqdm(p.imap(worker, files), total=len(files)))
