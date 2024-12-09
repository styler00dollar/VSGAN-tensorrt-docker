import sys

sys.path.append("/workspace/tensorrt/")
import vapoursynth as vs


core = vs.core
core.num_threads = 10  # can influence ram usage
core.std.LoadPlugin(path="/usr/local/lib/libvstrt.so")

"""
Experimental batch onnx test. Use core.trt normally like my other examples show for batch 1 onnx if
you just want to have stuff working.

Batching input can improve inference speed, but mlrt does not support such. A workaround is to
do batching within the model and output multiple frames at once. This seems to work well for compact.

Increased num_threads and num_streams in this example to have high gpu utilization.
"""


def inference_clip(video_path="", clip=None):
    clip = core.bs.VideoSource(source=video_path)
    clip = vs.core.resize.Bicubic(clip, format=vs.RGBH, matrix_in_s="709")

    stream0 = core.std.SelectEvery(clip, cycle=2, offsets=0)
    stream1 = core.std.SelectEvery(clip, cycle=2, offsets=1)

    prop = "test"
    clip = core.trt.Model(
        [stream0, stream1],
        engine_path="/workspace/tensorrt/2x_AnimeJaNai_V2_Compact_36k_op18_fp16_clamp_batch2.engine",
        num_streams=10,
        flexible_output_prop="test",
    )
    out0 = core.std.ShufflePlanes(
        [clip["clip"].std.PropToClip(prop=f"{prop}{i}") for i in range(3)],
        [0, 0, 0],
        vs.RGB,
    )
    out1 = core.std.ShufflePlanes(
        [clip["clip"].std.PropToClip(prop=f"{prop}{i}") for i in range(3, 6)],
        [0, 0, 0],
        vs.RGB,
    )

    clip = core.std.Interleave([out0, out1])

    clip = vs.core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s="709")
    return clip
