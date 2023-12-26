import numpy as np
import vapoursynth as vs
import onnxruntime as ort
from threading import Lock


def scene_detect(
    clip: vs.VideoNode,
    thresh: float = 0.98,
    onnx_path: str = "test.onnx",
    resolution: int = 256,
) -> vs.VideoNode:
    core = vs.core
    num_sessions = 3
    sessions = [
        ort.InferenceSession(
            onnx_path,
            providers=["TensorrtExecutionProvider"],
            trt_engine_cache_enable=True,
            trt_engine_cache_path="/workspace/tensorrt/",
            trt_fp16_enable=True,
            trt_max_workspace_size=7000000000,  # ~7gb
            trt_builder_optimization_level=5,
        )
        for _ in range(num_sessions)
    ]
    sessions_lock = [Lock() for _ in range(num_sessions)]

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

        fout = f[0].copy()
        I0 = frame_to_tensor(f[1])
        I1 = frame_to_tensor(f[2])
        in_sess = np.concatenate([I0, I1], axis=0)

        ort_session = sessions[local_index]
        result = ort_session.run(None, {"input": in_sess})[0][0][0]
        if result > thresh:
            fout.props._SceneChangeNext = 1
        return fout

    clip_down = clip.resize.Bicubic(
        resolution, resolution, format=vs.RGBH, matrix_in_s="709"
    )
    return core.std.ModifyFrame(clip, (clip, clip_down, clip_down[1:]), execute)
