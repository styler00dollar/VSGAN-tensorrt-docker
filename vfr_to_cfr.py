import glob
import os
from concurrent.futures import ThreadPoolExecutor

input_dir = "/"
output_dir = "/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def process_file(file_path):
    fixed_file_path = file_path.replace("'", "").replace("!", "")
    os.rename(file_path, fixed_file_path)

    out_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(fixed_file_path))[0] + "_mux.mkv"
    )
    os.system(
        f"ffmpeg -i '{file_path}' -map 0 -c:a copy -c:s copy -fps_mode cfr -crf 5 '{out_path}' -y"
    )


with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_file, files)
