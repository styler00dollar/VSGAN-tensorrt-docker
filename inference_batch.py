
import sys
import os
sys.path.append("/workspace/tensorrt/")
from inference_config import inference_clip

tmp_dir = "tmp/"
with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    video_path = f.readlines()[0]

clip = inference_clip(video_path)
clip.set_output()