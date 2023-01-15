import glob
import os
import multiprocessing
import shutil

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "/workspace/tensorrt/tmp"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

# creating folders if they dont exist
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def process_file(f):
    out_render_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_gmf_union.mkv"
    )

    os.system(
        f"vspipe -c y4m inference_config_gmf_union.py --arg source={f} - | ffmpeg -i pipe: -crf 10 -preset slow {out_render_path}"
    )


# processing gmf with multithreading since this drastically increases speed
pool = multiprocessing.Pool(2)
pool.map(process_file, files)
pool.close()
pool.join()

# processing with cugan in one loop since threading hurts performance
for f in files:
    # paths
    mux_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

    gmf_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_gmf_union.mkv"
    )

    os.system(f"vspipe /workspace/tensorrt/parse.py --arg source={gmf_path} -p .")

    # h264
    #os.system(
    #    f"vspipe -c y4m inference_config_cugan.py --arg source={gmf_path} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -scodec copy -crf 10 -preset slow {mux_path}"
    #)

    # av1_nvenc
    os.system(
       f"vspipe -c y4m inference_batch.py --arg source={gmf_path} - | ffmpeg -y -i {f} -thread_queue_size 100 -i pipe: -map 1 -map 0 -map -0:v -max_interleave_delta 0 -vcodec av1_nvenc -scodec copy -cq 20 -preset p2 {mux_path}"
    )

    #os.remove(gmf_path)
