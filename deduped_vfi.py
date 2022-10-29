# example usage: python main.py
# vapoursynth does not have audio support and processing multiple files is not really possible
# hacky script to make batch processing with audio and subtitle support
# make sure tmp_dir is also set in inference.py
# maybe should pass arguments instead of a text file instead
import glob
import os

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "tmp/"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

for f in files:
    # creating folders if they dont exist
    if os.path.exists(tmp_dir) == False:
        os.mkdir(tmp_dir)
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    # paths
    txt_path = os.path.join(tmp_dir, "tmp.txt")
    out_render_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_rendered.mkv"
    )
    mux_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    # writing filepath into temp txt
    # workaround to pass filename parameter
    f_txt = open(txt_path, "w")
    f_txt.write(str(f))
    f_txt.close()

    os.system("vspipe parse.py -p .")
    os.system(
        f"vspipe -c y4m ddfi.py - | ffmpeg -i pipe: -vcodec libsvtav1 -crf 20 {out_render_path}"
    )

    os.system(
        f"ffmpeg -y -loglevel error -i {f} -i {out_render_path}  -map 1 -map 0 -map -0:v -codec copy -max_interleave_delta 0 {mux_path}"
    )

    # deleting temp files
    os.remove(txt_path)
