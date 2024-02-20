import numpy as np
import vapoursynth as vs
import onnxruntime as ort
from threading import Lock

core = vs.core
ort.set_default_logger_severity(3)


def scene_detect(
    clip: vs.VideoNode,
    thresh: float = 0.98,
    onnx_path: str = "test.onnx",
    resolution: int = 256,
    num_sessions: int = 3,
    ssim_clip=None,
    ssim_thresh: float = 0.98,
) -> vs.VideoNode:
    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
    options = {}
    options["device_id"] = 0
    options["trt_engine_cache_enable"] = True
    options[
        "trt_timing_cache_enable"
    ] = True  # Using TensorRT timing cache to accelerate engine build time on a device with the same compute capability
    options["trt_engine_cache_path"] = "/workspace/tensorrt/"
    options["trt_fp16_enable"] = True
    options["trt_max_workspace_size"] = 7000000000  # ~7gb
    options["trt_builder_optimization_level"] = 5

    sessions = [
        ort.InferenceSession(
            onnx_path,
            providers=[
                ("TensorrtExecutionProvider", options),
                "CUDAExecutionProvider",
            ],
        )
        for _ in range(num_sessions)
    ]
    [Lock() for _ in range(num_sessions)]

    index = -1
    index_lock = Lock()

    def frame_to_tensor(frame: vs.VideoFrame):
        return np.stack(
            [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
        )

    def execute(n, f):
        nonlocal index
        with index_lock:
            index = (index + 1) % num_sessions
            local_index = index

        nonlocal ssim_clip
        nonlocal ssim_thresh
        if ssim_clip:
            ssim_clip = f[3].props.get("float_ssim")
            if ssim_clip and ssim_clip > ssim_thresh:
                return f[0].copy()

        fout = f[0].copy()
        I0 = frame_to_tensor(f[1])
        I1 = frame_to_tensor(f[2])
        in_sess = np.concatenate([I0, I1], axis=0)

        ort_session = sessions[local_index]
        result = ort_session.run(None, {"input": in_sess})[0][0][0]

        if result > thresh:
            fout.props._SceneChangeNext = 1
        else:
            fout.props._SceneChangeNext = 0
        return fout

    clip_down = clip.resize.Bicubic(
        resolution, resolution, format=vs.RGBH, matrix_in_s="709"
    )

    if ssim_clip:
        return core.std.ModifyFrame(
            clip=clip,
            clips=(clip, clip_down, clip_down[1:], ssim_clip),
            selector=execute,
        )

    return core.std.ModifyFrame(clip, (clip, clip_down, clip_down[1:]), execute)
