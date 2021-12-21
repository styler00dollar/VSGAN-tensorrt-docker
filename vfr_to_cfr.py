import glob
import os
input_dir = "/path/"
output_dir = "/path/"
files = glob.glob(input_dir + '/**/*.mkv', recursive=True)
files.sort()

for f in files:
    os.system(f"ffmpeg -i {f} -vsync cfr -crf 10 -c:a copy {os.path.join(output_dir, os.path.basename(f))}")
