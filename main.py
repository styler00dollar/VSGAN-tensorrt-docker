import glob
import os
import subprocess
from pathlib import Path

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "tmp/"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()


for file_path in files:
    relative_output_path = Path(os.path.dirname(file_path)).relative_to(input_dir)
    mux_output_dir = os.path.join(output_dir, relative_output_path)

    os.makedirs(mux_output_dir, exist_ok=True)

    out_render_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(file_path))[0] + "_rendered.mkv",
    )
    mux_path = os.path.join(
        mux_output_dir, os.path.splitext(os.path.basename(file_path))[0] + "_mux.mkv"
    )

    # only needed for dedup
    # os.system(f"vspipe /workspace/tensorrt/parse.py --arg source='{input_file_path}' -p .")

    """
    Examples: 
    -vcodec libx265 -crf 10 -preset slow
    -vcodec av1_nvenc -cq 1 -preset p7 -multipass 2 -tune hq -highbitdepth 1
    -vcodec libsvtav1

    ffmpeg -encoders: will show all encoders
    ffmpeg -h encoder=libsvtav1: will show all options of one encoder
    """

    ps = subprocess.Popen(
        [
            "vspipe",
            "-c",
            "y4m",
            "inference_batch.py",
            "--arg",
            f"source={file_path}",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    output = subprocess.check_output(
        [
            "ffmpeg",
            "-i",
            f"{file_path}",
            "-thread_queue_size",
            "100",
            "-i",
            "pipe:",
            "-map",
            "1",
            "-map",
            "0",
            "-map",
            "-0:v",
            "-max_interleave_delta",
            "0",
            "-scodec",
            "copy",
            "-vcodec",
            "libsvtav1",
            "-svtav1-params",
            "tune=0:fast-decode=1:auto-tiling=1:enable-dlf=3",
            f"{mux_path}",
            "-y",
        ],
        stdin=ps.stdout,
    )
    ps.wait()
