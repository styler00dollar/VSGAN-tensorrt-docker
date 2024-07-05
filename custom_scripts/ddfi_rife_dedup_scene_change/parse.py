import sys

sys.path.append("/workspace/tensorrt/")

import vapoursynth as vs
import functools
from src.scene_detect import scene_detect

core = vs.core
core.num_threads = 4
core.max_cache_size = 4096 * 8

core.std.LoadPlugin(path="/usr/local/lib/libfmtconv.so")

import sys


# https://github.com/xyx98/my-vapoursynth-script/blob/master/xvs.py
def props2csv(
    clip: vs.VideoNode,
    props: list,
    titles: list,
    output="info.csv",
    sep="\t",
    charset="utf-8",
    tostring=None,
):
    file = open(output, "w", encoding=charset)
    file.write(sep.join(["n"] + titles))
    tostring = (
        tostring
        if callable(tostring)
        else lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
    )

    def tocsv(n, f, clip):
        file.write(
            "\n"
            + sep.join(
                [str(n)]
                + [tostring(eval("f.props." + i, globals(), {"f": f})) for i in props]
            )
        )

        return clip
        file.close()

    return core.std.FrameEval(clip, functools.partial(tocsv, clip=clip), prop_src=clip)


clip = core.bs.VideoSource(source=globals()["source"])

clip = scene_detect(
    clip,
    fp16=True,
    thresh=0.85,
    model=12,
)

offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
offs1 = core.std.CopyFrameProps(offs1, clip)
offs1 = core.vmaf.Metric(clip, offs1, 2)
offs1 = core.std.MakeDiff(offs1, clip)
offs1 = core.fmtc.bitdepth(offs1, bits=16)
offs1 = core.std.Expr(offs1, "x 32768 - abs")
offs1 = core.std.PlaneStats(offs1)

offs1 = props2csv(
    offs1,
    props=["_AbsoluteTime", "float_ssim", "PlaneStatsMax", "_SceneChangeNext"],
    output="/workspace/tensorrt/tmp/infos_running.txt",
    titles=["_AbsoluteTime", "float_ssim", "PlaneStatsMax", "_SceneChangeNext"],
)
offs1.set_output()
