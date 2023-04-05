import sys
import os

sys.path.append("/workspace/tensorrt/")
from inference_config import inference_clip

clip = inference_clip(
    globals()["source"],
)
clip.set_output()
