# example usage: python main.py
# hacky script to make batch processing with audio and subtitle support
import glob
import os
import shutil

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "tmp/"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

for f in files:
    # creating folders if they dont exist
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

    # paths
    out_render_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_rendered.mkv"
    )
    mux_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    # only needed for dedup
    # os.system(f"vspipe /workspace/tensorrt/parse.py --arg source='{f}' -p .")

    # x265
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libx265 -crf 10 -preset slow '{mux_path}'"
    # )

    # aom av1
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libaom-av1 -cpu-used 6 -lag-in-frames 48 -arnr-max-frames 4 -arnr-strength 1 -enable-cdef 1 -enable-restoration 0 -tune ssim -aom-params film-grain-table='/workspace/tensorrt/grain.tbl',input-depth=10,fp-mt=1,keyint=240 -crf 10 -tile-columns 1 -tile-rows 1 -row-mt 1 '{mux_path}'"
    # )

    ### medium ###
    # x265
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libx265 -crf 10 '{mux_path}'"
    # )

    ### faster ###
    # x264
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -crf 10 '{mux_path}'"
    # )

    # x264
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -crf 10 -preset slow '{mux_path}'"
    # )

    # svt av1
    os.system(
        f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libsvtav1 -svtav1-params tune=0:enable-overlays=1:enable-qm=1:tile-columns=1:enable-restoration=0:fast-decode=1 -preset 8 -crf 10 -tune fastdecode '{mux_path}'"
    )

    # av1_nvenc
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i '{f}' -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -vcodec av1_nvenc -scodec copy -cq 1 -preset p7 -multipass 2 -tune hq -highbitdepth 1 '{mux_path}'"
    # )

    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source='{f}' - | ffmpeg -y -i pipe: %05d.png"
    # )

    # deleting temp files
    # os.remove(txt_path)
