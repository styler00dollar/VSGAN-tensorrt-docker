import sys

sys.path.append("/workspace/tensorrt/")

import vapoursynth as vs
import os
from src.rife_trt import rife_trt
import pandas as pd
import math
import functools

core = vs.core

core.std.LoadPlugin(path="/usr/local/lib/x86_64-linux-gnu/vapoursynth/libvfrtocfr.so")
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")


ssimt = 0.999
pxdifft = 10240
consecutivet = 2
core.num_threads = 4
core.max_cache_size = 4096

tmp_dir = "tmp/"

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def metrics_func(clip):
    offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
    offs1 = core.std.CopyFrameProps(offs1, clip)
    return core.vmaf.Metric(clip, offs1, 2)


# frames to delete
def processInfo():
    with open(os.path.join(tmp_dir, "infos_running.txt"), "r") as f:
        lines = [i.split("\t") for i in f][1:]
    for i in range(len(lines)):
        lines[i][0] = int(lines[i][0])  # _AbsoluteTime
        lines[i][1] = int(float(lines[i][1]) * 1000)  # float_ssim
        lines[i][2] = float(lines[i][2])  # PlaneStatsMax
        lines[i][3] = int(lines[i][3])  # _SceneChangeNext
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

clip = core.bs.VideoSource(source=globals()["source"])


def CsvToProp(clip: vs.VideoNode, csv_file: str) -> vs.VideoNode:
    df = pd.read_csv(csv_file, header="infer", sep="\s+|\t+|\s+\t+|\t+\s+")

    def csv_to_prop(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f.copy()
        row = df.iloc[n]

        """
        0                   n
        1       _AbsoluteTime
        2          float_ssim
        3       PlaneStatsMax
        4    _SceneChangeNext
        """

        fout.props["_SceneChangeNext"] = row["_SceneChangeNext"]
        fout.props["float_ssim"] = row["float_ssim"]

        return fout

    return clip.std.ModifyFrame(clips=clip, selector=csv_to_prop)


clip = CsvToProp(clip, "tmp/infos_running.txt")

clip = core.std.DeleteFrames(clip, dels)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBS)
clip_orig = vs.core.std.Interleave([clip] * 8)

clip = rife_trt(
    clip,
    multi=8,
    scale=1.0,
    device_id=0,
    num_streams=2,
    engine_path="/workspace/tensorrt/rife418_v2_ensembleFalse_op20_clamp_onnxslim.engine",
)
clip = core.akarin.Select([clip, clip_orig], clip, "x._SceneChangeNext 1 0 ?")
clip = core.akarin.Select([clip, clip_orig], clip, "x.float_ssim 0.999 >")


clip = core.vfrtocfr.VFRToCFR(
    clip, os.path.join(tmp_dir, "tsv2nX8.txt"), 192000, 1001, True
)  # 24fps * 8

# https://forum.doom9.org/showthread.php?t=171417
# reduce from 192 to 60 fps, skipping frames to make inference faster
target_fps_num = 60
target_fps_den = 1


def frame_adjuster(n, clip, target_fps_num, target_fps_den):
    real_n = math.floor(
        n / (target_fps_num / target_fps_den * clip.fps_den / clip.fps_num)
    )
    one_frame_clip = clip[real_n] * (len(clip) + 100)
    return one_frame_clip


clip = core.std.BlankClip(
    clip,
    length=math.floor(
        len(clip) * target_fps_num / target_fps_den * clip.fps_den / clip.fps_num
    ),
    fpsnum=target_fps_num,
    fpsden=target_fps_den,
)
clip = core.std.FrameEval(
    clip,
    functools.partial(
        frame_adjuster,
        clip=clip,
        target_fps_num=target_fps_num,
        target_fps_den=target_fps_den,
    ),
)

clip = vs.core.resize.Bicubic(clip, format=vs.RGBH)

clip = core.trt.Model(
    clip,
    engine_path="/workspace/tensorrt/2x_AnimeJaNai_V2.1_SmoothRC12_Compact_34k_clamp_fp16_op20.engine",
    num_streams=2,
)
clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")

clip.set_output()
