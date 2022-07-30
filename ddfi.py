import vapoursynth as vs
import os

core = vs.core

core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libmvtools.so")
core.std.LoadPlugin(path="/usr/local/lib/x86_64-linux-gnu/vapoursynth/libvfrtocfr.so")

ssimt = 0.999
pxdifft = 10240
consecutivet = 2
core.num_threads = 4
core.max_cache_size = 4096

tmp_dir = "tmp/"

# frames to delete
def processInfo():
    with open(os.path.join(tmp_dir, "infos_running.txt"), "r") as f:
        lines = [i.split("\t") for i in f][1:]
    for i in range(len(lines)):
        lines[i][0] = int(lines[i][0])
        lines[i][1] = int(float(lines[i][1]) * 1000)
        lines[i][2] = float(lines[i][2])
        lines[i][3] = int(lines[i][3])
    lines.sort()
    startpts = lines[0][1]
    consecutive = 0

    dels = []
    tsv2o = []
    for i in range(len(lines)):
        l = lines[i]
        if l[2] >= ssimt and l[3] <= pxdifft and consecutive < consecutivet:
            consecutive += 1
            dels.append(l[0])
        else:
            consecutive = 0
            tsv2o.append(l[1] - startpts)
    return dels, tsv2o


def newTSgen(tsv2o):
    ts_new = list()
    outfile = open(os.path.join(tmp_dir, "tsv2nX8.txt"), "w", encoding="utf-8")
    ts_o = [i for i in tsv2o][1:]

    for x in range(len(ts_o) - 1):
        ts_new.append(str(float(ts_o[x])))
        for i in range(1, 8):
            ts_new.append(
                str(float(ts_o[x]) + (float(ts_o[x + 1]) - float(ts_o[x])) / 8 * i)
            )
    print("#timestamp format v2", file=outfile)
    for x in range(len(ts_new)):
        print(ts_new[x], file=outfile)
    print(ts_o[len(ts_o) - 1], file=outfile)
    outfile.close()


dels, tsv2o = processInfo()
newTSgen(tsv2o)

with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    video_path = f.readlines()[0]
clip = core.ffms2.Source(video_path, cachefile="ffindex")
clip = core.std.DeleteFrames(clip, dels)
sup = core.mv.Super(clip, pel=1, levels=1)
bw = core.mv.Analyse(sup, isb=True, levels=1, truemotion=False)
clip = core.mv.SCDetection(clip, bw, thscd1=200, thscd2=85)
clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=1)
clip = core.rife.RIFE(clip, model=11, sc=True, skip=False, multiplier=8)
clip = core.resize.Bicubic(
    clip, format=vs.YUV420P10, matrix=1, dither_type="error_diffusion"
)
clip = core.vfrtocfr.VFRToCFR(
    clip, os.path.join(tmp_dir, "tsv2nX8.txt"), 192000, 1001, True
)  # 24fps * 8
sup = core.mv.Super(clip)
fw = core.mv.Analyse(sup)
bw = core.mv.Analyse(sup, isb=True)
clip = core.mv.FlowFPS(clip, sup, bw, fw, 60, 1)
clip.set_output()
