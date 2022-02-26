from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="input.mkv", help="input file name")
parser.add_argument("-s", "--subs", type=str, default="ass", help="subtitle file format")
parser.add_argument("-a", "--audio", type=str, default="aac", help="audio file format")
parser.add_argument("-c", "--cache", type=str, default="250MiB", help="amount of cache")
args = parser.parse_args()

tmp_dir = "tmp/"

# creating folders if they dont exist
if os.path.exists(tmp_dir) == False:
    os.mkdir(tmp_dir)

# paths
txt_path = os.path.join(tmp_dir, "tmp.txt")
subs_path = os.path.join(tmp_dir, f"subs.{args.subs}")
audio_path = os.path.join(tmp_dir, f"audio.{args.audio}") # ogg, aac, flac, ac3

# writing filepath into temp txt
# workaround to pass filename parameter
f_txt = open(txt_path, "w")
f_txt.write(str(args.input))
f_txt.close()

# calling vspipe and piping into ffmpeg
###############################################
# AUDIO
# -map 0:2 means second audio track

# copy audio without reencoding
os.system(f"ffmpeg -i {args.input} -vn -acodec copy -c copy {audio_path}")
# reencode if extract fails
#os.system(f"ffmpeg -i {f} -map 0:2 -vn {audio_path}")
###############################################
# extract subtitles, -map 0:s:1 means second subtitle track
#os.system(f"ffmpeg -i {args.input} -map 0:s:0 -c:s copy {subs_path}")

# no subs
os.system(f"vspipe -c y4m inference_batch.py - | mpv - --audio-file={audio_path} --demuxer-max-bytes={args.cache}")
# subs
#os.system(f"vspipe -c y4m inference_batch.py - | mpv - --audio-file={audio_path} --sub-files={subs_path} --demuxer-max-bytes={args.cache}")

# deleting temp files
os.remove(txt_path)
os.remove(subs_path)
os.remove(audio_path)