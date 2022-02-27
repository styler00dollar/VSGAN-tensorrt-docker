import glob
import os
input_dir = "/path/"
output_dir = "/path/"
files = glob.glob(input_dir + '/**/*.mkv', recursive=True)
files.sort()

audio_path = "tmp/subs.flac"
subs_path = "tmp/audio.ass"

for f in files:
    tmp_render_path = os.path.join(output_dir, os.path.basename(f))
    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv")

    # extract audio and subs, specify your own tracks ids
    # ffprobe -i video.mkv
    os.system(f"ffmpeg -i {f} -map 0:2 -acodec copy {audio_path}")
    os.system(f"ffmpeg -i {f} -map 0:4 {subs_path}")
    # render video without audio and subtitles
    os.system(f"ffmpeg -i {f} -an -sn -vsync cfr -crf 15 -preset slow {tmp_render_path}")
    # merge
    os.system(f"ffmpeg -i {subs_path} -c:s mov_text -i {tmp_render_path} -i {audio_path} -c copy {out_path}")
    # delete temp files
    os.remove(tmp_render_path)
    os.remove(audio_path)
    os.remove(subs_path)
    
# only render video into cfr without audio and subtitles
"""
for f in files:
    os.system(f"ffmpeg -i {f} -vsync cfr -crf 10 -c:a copy {os.path.join(output_dir, os.path.basename(f))}")
"""
