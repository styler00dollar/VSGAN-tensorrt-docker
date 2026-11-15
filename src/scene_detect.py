from __future__ import annotations

import os
from dataclasses import dataclass
from queue import Queue

import numpy as np
import onnxruntime as ort
import vapoursynth as vs

from .download import check_and_download

core = vs.core
ort.set_default_logger_severity(3)

MODEL_DIR = "/workspace/tensorrt/models"
TRT_CACHE_DIR = "/workspace/tensorrt"


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    width: int
    height: int
    # Frames fed to the model, relative to the frame being classified.
    # e.g. (0, 1) = current + next; (-2, -1, 0, 1, 2) = a 5-frame window.
    frame_offsets: tuple[int, ...] = (0, 1)
    # 0 -> frames concatenated along the channel axis (6 CHW for 2 frames)
    # 1 -> frames interleaved as a frame axis ((3, N, H, W) for N frames)
    stack_axis: int = 0
    frame_format: int = vs.RGBH
    # How frames past the clip edges are filled: "blank" or "repeat".
    edge_fill: str = "blank"
    # Which ONNX output tensor to read.
    output_index: int = 0
    # Index into that tensor. (0, 0) for a single scalar / (1, 1) output;
    # (0, 2) selects the middle frame of a (1, 5) score vector.
    score_index: tuple[int, ...] = (0, 0)


MODELS: dict[int, ModelSpec] = {
    0: ModelSpec(
        "sc_efficientnetv2b0_17957_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx",
        256, 256, edge_fill="repeat",
    ),  # ~0.85 thresh
    1: ModelSpec(
        "sc_efficientnetv2b0+rife46_flow_1362_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx",
        256, 256, edge_fill="repeat",
    ),  # ~0.85 thresh
    2: ModelSpec(
        "sc_swinv2_small_window16_10412_256_CHW_6ch_clamp_softmax_op17_fp16.onnx",
        256, 256, edge_fill="repeat",
    ),  # ~0.85 thresh
    3: ModelSpec(
        "sc_swinv2_small_window16+rife46_flow_1814_256_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx",
        256, 256, edge_fill="repeat",
    ),  # ~0.85 thresh
    4: ModelSpec(
        "sc_maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx",
        224, 224, edge_fill="repeat",
    ),  # ~0.85 thresh
    5: ModelSpec(
        "sc_autoshot_clamp_op17_5img.onnx",
        48, 27,
        frame_offsets=(-2, -1, 0, 1, 2),
        stack_axis=1,
        frame_format=vs.RGBS,
        score_index=(0, 2),  # middle frame of the 5-frame window
    ),  # ~0.2 thresh
}


class SessionPool:
    def __init__(self, sessions: list[ort.InferenceSession]) -> None:
        self._queue: Queue[ort.InferenceSession] = Queue()
        for session in sessions:
            self._queue.put(session)

    def run(self, inputs: np.ndarray) -> list[np.ndarray]:
        session = self._queue.get()
        try:
            return session.run(None, {"input": inputs})
        finally:
            self._queue.put(session)


def _tensorrt_options(fp16: bool) -> dict:
    return {
        "device_id": 0,
        "trt_engine_cache_enable": True,
        "trt_timing_cache_enable": True,
        "trt_engine_cache_path": TRT_CACHE_DIR,
        "trt_fp16_enable": fp16,
        "trt_max_workspace_size": 7_000_000_000,
        "trt_builder_optimization_level": 5,
    }


def _create_session_pool(
    onnx_path: str, fp16: bool, num_sessions: int
) -> SessionPool:
    if num_sessions < 1:
        raise ValueError("num_sessions must be at least 1")

    options = _tensorrt_options(fp16)
    sessions = [
        ort.InferenceSession(
            onnx_path,
            #providers=[("TensorrtExecutionProvider", dict(options))],
            providers=["CUDAExecutionProvider"]
        )
        for _ in range(num_sessions)
    ]
    return SessionPool(sessions)


def _frame_to_tensor(frame: vs.VideoFrame) -> np.ndarray:
    return np.stack(
        [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
    )


def _build_input(frames: list[vs.VideoFrame], spec: ModelSpec) -> np.ndarray:
    window = [
        _frame_to_tensor(frames[1 + i])
        for i in range(len(spec.frame_offsets))
    ]

    if spec.stack_axis == 0:
        # Concatenate frames along the channel axis: (3*N, H, W)
        return np.concatenate(window, axis=0)

    if spec.stack_axis == 1:
        # Keep a frame axis: (3, N, H, W)
        return np.stack(window, axis=1)

    raise ValueError(f"Unsupported stack_axis: {spec.stack_axis}")


def _pad(
    clip: vs.VideoNode, length: int, fill: str, at_end: bool
) -> vs.VideoNode:
    if fill == "blank":
        return core.std.BlankClip(clip, length=length)
    if fill == "repeat":
        source = clip[clip.num_frames - 1:] if at_end else clip[:1]
        return core.std.Loop(source, times=length)
    raise ValueError(f"Unsupported edge_fill: {fill}")


def _shift(
    clip: vs.VideoNode, offset: int, fill: str = "blank"
) -> vs.VideoNode:
    """Shift a clip by offset frames, filling edges."""
    if offset == 0:
        return clip

    n = clip.num_frames
    pad = min(abs(offset), n)

    if offset > 0:
        remaining = clip[offset:] if offset < n else None
        tail = _pad(clip, pad, fill, at_end=True)
        return remaining + tail if remaining is not None else tail

    remaining = clip[: n + offset] if n + offset > 0 else None
    head = _pad(clip, pad, fill, at_end=False)
    return head + remaining if remaining is not None else head


def _extract_score(outputs: list[np.ndarray], spec: ModelSpec) -> float:
    return float(np.asarray(outputs[spec.output_index])[spec.score_index].item())


def scene_detect(
    clip: vs.VideoNode,
    thresh: float = 0.85,
    model: int = 4,
    fp16: bool = True,
    num_sessions: int = 2,
) -> vs.VideoNode:
    spec = MODELS.get(model)
    if spec is None:
        raise ValueError(f"Unknown scene-detect model: {model}")

    onnx_path = os.path.join(MODEL_DIR, spec.filename)
    check_and_download(onnx_path)

    pool = _create_session_pool(onnx_path, fp16, num_sessions)

    clip_down = clip.resize.Bicubic(
        spec.width,
        spec.height,
        format=spec.frame_format,
        matrix_in_s="709",
    )

    frames = (
        clip,
        *(_shift(clip_down, offset, spec.edge_fill)
          for offset in spec.frame_offsets),
    )

    def execute(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        out = f[0].copy()

        inputs = _build_input(f, spec)
        score = _extract_score(pool.run(inputs), spec)

        out.props._sceneFloat = score
        out.props._SceneChangeNext = int(score > thresh)
        return out

    return core.std.ModifyFrame(clip, list(frames), execute)
