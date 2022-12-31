import vapoursynth as vs
import os
import functools

core = vs.core
core.num_threads = 32
core.max_cache_size = 4096*8

core.std.LoadPlugin(path="/usr/lib/x86_64-linux-gnu/libffms2.so")
core.std.LoadPlugin(path="/usr/local/lib/libfmtconv.so")

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
        else lambda x: x.decode("utf-8")
        if isinstance(x, bytes)
        else str(x)
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


tmp_dir = "tmp/"
with open(os.path.join(tmp_dir, "tmp.txt")) as f:
    video_path = f.readlines()[0]
clip = core.ffms2.Source(video_path, cache=False)
offs1 = core.std.BlankClip(clip, length=1) + clip[:-1]
offs1 = core.std.CopyFrameProps(offs1, clip)
offs1 = core.vmaf.Metric(clip, offs1, 2)
offs1 = core.std.MakeDiff(offs1, clip)
offs1 = core.fmtc.bitdepth(offs1, bits=16)
offs1 = core.std.Expr(offs1, "x 32768 - abs")
offs1 = core.std.PlaneStats(offs1)
offs1 = props2csv(
    offs1,
    props=["_AbsoluteTime", "float_ssim", "PlaneStatsMax"],
    output=os.path.join(tmp_dir, "infos_running.txt"),
    titles=[],
)
offs1.set_output()
