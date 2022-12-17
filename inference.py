import sys

sys.path.append("/workspace/tensorrt/")
from inference_config import inference_clip

video_path = "test.mp4"
clip = inference_clip(video_path)
clip.set_output()
