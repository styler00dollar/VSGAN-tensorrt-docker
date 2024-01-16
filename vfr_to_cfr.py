import glob
import os

input_dir = "/"
output_dir = "/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

if os.path.exists(output_dir) == False:
    os.mkdir(output_dir)

for f in files:
    out_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    os.system(
        f"ffmpeg -i '{f}' -map 0 -c:v libx264 -c:a copy -c:s copy -fps_mode cfr -crf 5 -preset medium '{out_path}' -y"
    )
