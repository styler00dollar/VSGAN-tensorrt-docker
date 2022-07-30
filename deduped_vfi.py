import os 
import glob 

#subprocess.run(["vspipe", "parse.vpy", "-p", "."])

input_dir = "/workspace/tensorrt/input/"
tmp_dir = "tmp/"
output_dir = "/workspace/tensorrt/output/"

files = glob.glob(input_dir + "/**/*.mp4", recursive=True)
files.sort()


for f in files:
    # creating folders if they dont exist
    if os.path.exists(tmp_dir) == False:
        os.mkdir(tmp_dir)
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    # paths
    txt_path = os.path.join(tmp_dir, "tmp.txt")
    subs_path = os.path.join(tmp_dir, "subs.ass")  # srt, ass
    audio_path = os.path.join(tmp_dir, "audio.flac")  # ogg, aac, flac, ac3
    out_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(f))[0] + "_mux.mkv"
    )

    # delete cache file if exists
    if os.path.exists(os.path.join(tmp_dir, "ffindex")) == True:
        os.remove(os.path.join(tmp_dir, "ffindex"))
    if os.path.exists(os.path.join(tmp_dir, "infos_running.txt")) == True:
        os.remove(os.path.join(tmp_dir, "infos_running.txt"))
    if os.path.exists(os.path.join(tmp_dir, "tsv2nX8.txt")) == True:
        os.remove(os.path.join(tmp_dir, "tsv2nX8.txt"))
    if os.path.exists(audio_path) == True:
        os.remove(audio_path)
    if os.path.exists(subs_path) == True:
        os.remove(subs_path)
    if os.path.exists(audio_path) == True:
        os.remove(audio_path)

    # writing filepath into temp txt
    # workaround to pass filename parameter
    f_txt = open(txt_path, "w")
    f_txt.write(str(f))
    f_txt.close()

    # calling vspipe and piping into ffmpeg
    ###############################################
    # AUDIO
    # -map 0:2 means second audio track

    # copy audio without reencoding
    os.system(f"ffmpeg -i {f} -vn -acodec copy {audio_path}")
    # reencode if extract fails
    # os.system(f"ffmpeg -i {f} -vn {audio_path}")
    ###############################################
    # extract subtitles, -map 0:s:1 means second subtitle track
    os.system(f"ffmpeg -i {f} -map 0:s:0 {subs_path}")


    os.system("vspipe parse.vpy -p .")
    os.system(f"vspipe -c y4m ddfi.py - | ffmpeg -i {subs_path} -c:s mov_text -i pipe: -preset slow {out_path} -i {audio_path} -c copy")
    