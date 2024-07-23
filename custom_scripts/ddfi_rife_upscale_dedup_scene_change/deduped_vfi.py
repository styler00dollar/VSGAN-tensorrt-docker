import glob
import os

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "tmp/"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

for f in files:
    # creating folders if they dont exist
    if os.path.exists(tmp_dir) is False:
        os.mkdir(tmp_dir)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    mux_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    # metrics
    os.system(f"vspipe parse.py --arg source='{f}' -p .")

    # render
    os.system(
        f"vspipe -c y4m ddfi.py --arg source='{f}' -  | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libsvtav1 -svtav1-params tune=0,enable-overlays=1,enable-qm=1 -preset 6 -crf 10 '{mux_path}'"
    )
