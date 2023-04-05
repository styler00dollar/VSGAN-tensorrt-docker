# example usage: python main.py
# vapoursynth does not have audio support and processing multiple files is not really possible
# hacky script to make batch processing with audio and subtitle support
# make sure tmp_dir is also set in inference.py
# maybe should pass arguments instead of a text file instead
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
    # os.system(f"vspipe /workspace/tensorrt/parse.py --arg source={f} -p .")

    # os.system(f"vspipe -c y4m inference_batch.py - | ffmpeg -i pipe: {out_render_path}")
    # os.system(
    #    f"ffmpeg -y -loglevel error -i {f} -i {out_render_path}  -map 1 -map 0 -map -0:v -codec copy -max_interleave_delta 0 {mux_path}"
    # )

    ###### example presets ######
    # speeds are resizised 4k footage with 7950x
    # in terms of color banding:
    # (less) x264 >  libx265 > libaom-av1 (with grain) > libsvtav1 > av1_nvenc (more)
    # new encoders might result in smaller filesize/quality ratio, but there are banding problems which persist even with higher quality settings though
    # in my personal opinion, stick with x264 for no banding and speed and aom for quality for filesize

    ### slower ###
    # x265 crf10 preset slow [4fps]
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libx265 -crf 10 -preset slow {mux_path}"
    # )

    # aom av1 (for quality/filesize but slow, encoder has banding issues without grain table) [3fps]
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libaom-av1 -cpu-used 6 -lag-in-frames 48 -arnr-max-frames 4 -arnr-strength 1 -enable-cdef 1 -enable-restoration 0 -tune ssim -aom-params film-grain-table='/workspace/tensorrt/grain.tbl',input-depth=10,fp-mt=1,keyint=240 -crf 10 -tile-columns 1 -tile-rows 1 -row-mt 1 {mux_path}"
    # )

    ### medium ###
    # x265 crf10 default preset [13fps]
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libx265 -crf 10 {mux_path}"
    # )

    ### faster ###
    # x264 crf10 default preset [43fps]
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -crf 10 {mux_path}"
    # )

    # x264 crf10 preset slow [31fps]
    os.system(
        f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -crf 10 -preset slow {mux_path}"
    )

    # svt av1 (encoder has banding issues) [38fps]
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -vcodec libsvtav1 -svtav1-params tune=0,enable-overlays=1,enable-qm=1 -preset 8 -crf 10 {mux_path}"
    # )

    # av1_nvenc (only rtx4000, this hw encoder has banding issues. High bit depth may help. -qp for further filesize adjustment) [54fps]
    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -vcodec av1_nvenc -scodec copy -cq 1 -preset p7 -multipass 2 -tune hq -highbitdepth 1 {mux_path}"
    # )

    # os.system(
    #   f"vspipe -c y4m inference_batch.py --arg source={f} - | ffmpeg -y -i pipe: %05d.png"
    # )

    # deleting temp files
    # os.remove(txt_path)
