import glob
import os

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "tmp/"
output_dir = "/workspace/tensorrt/output/"
files = glob.glob(input_dir + "/**/*.mkv", recursive=True)
files.sort()

# creating folders if they dont exist
if os.path.exists(tmp_dir) == False:
    os.mkdir(tmp_dir)
if os.path.exists(output_dir) == False:
    os.mkdir(output_dir)

for f in files:
    # paths
    subs_path = os.path.join(tmp_dir, "subs.ass")  # srt, ass
    audio_path = os.path.join(tmp_dir, "audio.flac")  # ogg, aac, flac, ac3

    tmp_render_path = os.path.join(tmp_dir, os.path.basename(f))
    out_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    # extract audio and subs, specify your own tracks ids
    # ffprobe -i video.mkv
    # os.system(f"ffmpeg -i {f} -map 0:0 -acodec copy {audio_path}")
    os.system(f"ffmpeg -i {f} -vn {audio_path}")

    # os.system(f"ffmpeg -i {f} -map 0:0 {subs_path}")
    os.system(f"ffmpeg -i {f} {subs_path}")

    # render video without audio and subtitles
    os.system(
        f"ffmpeg -i {f} -an -sn -vsync cfr -crf 15 -preset slow {tmp_render_path}"
    )
    # merge
    os.system(
        f"ffmpeg -i {subs_path} -c:s mov_text -i {tmp_render_path} -i {audio_path} -c copy {out_path}"
    )

    # delete temp files
    if os.path.exists(subs_path) == True:
        os.remove(subs_path)
    if os.path.exists(audio_path) == True:
        os.remove(audio_path)
    if os.path.exists(tmp_render_path) == True:
        os.remove(tmp_render_path)

# only render video into cfr without audio and subtitles
"""
for f in files:
    os.system(f"ffmpeg -i {f} -vsync cfr -crf 10 -c:a copy {os.path.join(output_dir, os.path.basename(f))}")
"""
