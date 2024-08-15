import numpy as np
import vapoursynth as vs
import onnxruntime as ort
from threading import Lock
from .download import check_and_download
import os

core = vs.core
ort.set_default_logger_severity(3)


def scene_detect(
    clip: vs.VideoNode,
    thresh: float = 0.93,
    model: int = 0,
    fp16: bool = True,
    num_sessions: int = 2,
    ssim_clip=None,
    ssim_thresh: float = 0.98,
) -> vs.VideoNode:
    #######################################
    # own models, everything experimental #
    #######################################
    # requires ~0.8-0.9 thresh
    if model == 0:
        onnx_path = (
            "sc_efficientformerv2_s0_12263_224_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx"
        )
        resolution = [224, 224]
        onnx_type = "2img"
    elif model == 1:
        onnx_path = "sc_efficientformerv2_s0+rife46_flow_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx"
        resolution = [224, 224]
        onnx_type = "2img"
    elif model == 2:
        onnx_path = (
            "sc_efficientnetv2b0_17957_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx"
        )
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 3:
        onnx_path = "sc_efficientnetv2b0+rife46_flow_1362_256_CHW_6ch_clamp_softmax_op17_fp16_sim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 4:
        onnx_path = (
            "sc_swinv2_small_window16_10412_256_CHW_6ch_clamp_softmax_op17_fp16.onnx"
        )
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 5:
        onnx_path = "sc_swinv2_small_window16+rife46_flow_1814_256_84119_224_CHW_6ch_clamp_softmax_op17_fp16.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 6:
        onnx_path = "sc_tf_efficientnetv2_b0.in1k_48x27_b100_30k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
        resolution = [48, 27]
        onnx_type = "2img"
    elif model == 7:
        onnx_path = "sc_davit_small.msft_in1k_256px_b100_30k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 8:
        onnx_path = "sc_davit_small.msft_in1k_256px_b100_40k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 9:
        onnx_path = "sc_maxxvitv2_nano_rw_256.sw_in1k_256px_b100_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 10:
        onnx_path = "sc_maxxvitv2_nano_rw_256.sw_in1k_256px_b100_30k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    elif model == 11:
        onnx_path = "sc_maxxvitv2_rmlp_base_rw_224.sw_in12k_b80_224px_20k_coloraug0.4_6ch_clamp_softmax_fp16_op17_onnxslim.onnx"
        resolution = [224, 224]
        onnx_type = "2img"
    elif model == 12:
        onnx_path = "sc_mobilevitv2_050.cvnets_in1k+efficientvit_b2.r288_in1k_rife422_sobel_256px_5k_CHW_6ch_clamp_softmax_op20_fp16_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    ################
    # other models #
    ################
    # requires ~0.2 thresh
    elif model == 13:
        onnx_path = "sc_autoshot_clamp_op17_5img.onnx"
        resolution = [48, 27]
        onnx_type = "5img"
    # requires ~0.45 thresh
    elif model == 14:
        onnx_path = "sc_shift_lpips_alex_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    # requires ~0.45 thresh
    elif model == 15:
        onnx_path = "sc_shift_lpips_vgg_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"
    # requires ~0.25 thresh
    elif model == 16:
        onnx_path = "sc_dists_256px_CHW_6ch_clamp_op20_fp16_onnxslim.onnx"
        resolution = [256, 256]
        onnx_type = "2img"

    onnx_path = os.path.join("/workspace/tensorrt/models/", onnx_path)
    check_and_download(onnx_path)

    # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
    options = {}
    options["device_id"] = 0
    options["trt_engine_cache_enable"] = True
    options["trt_timing_cache_enable"] = (
        True  # Using TensorRT timing cache to accelerate engine build time on a device with the same compute capability
    )
    options["trt_engine_cache_path"] = (
        "/workspace/tensorrt"  # "/home/user/Schreibtisch/VSGAN-tensorrt-docker/"
    )
    options["trt_fp16_enable"] = fp16
    options["trt_max_workspace_size"] = 7000000000  # ~7gb
    options["trt_builder_optimization_level"] = 5

    sessions = [
        ort.InferenceSession(
            onnx_path,
            providers=[
                ("TensorrtExecutionProvider", options),
                # "CUDAExecutionProvider",
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

        if onnx_type == "2img":
            in_sess = np.concatenate([I0, I1], axis=0)
        elif onnx_type == "5img":
            I2 = frame_to_tensor(f[3])
            I3 = frame_to_tensor(f[4])
            I4 = frame_to_tensor(f[5])
            in_sess = np.stack([I0, I1, I2, I3, I4], axis=1)

        ort_session = sessions[local_index]

        if model == 16:
            result = ort_session.run(None, {"input": in_sess})[0]
        else:
            result = ort_session.run(None, {"input": in_sess})[0][0]

        if model != 16:
            if onnx_type == "2img":
                result = result[0]
            elif onnx_type == "5img":
                result = result[2]

        fout.props._sceneFloat = float(result)

        if result > thresh:
            fout.props._SceneChangeNext = 1
        else:
            fout.props._SceneChangeNext = 0
        return fout

    if onnx_type == "5img":
        clip_down = clip.resize.Bicubic(
            resolution[0], resolution[1], format=vs.RGBS, matrix_in_s="709"
        )

        shift_up2 = clip_down.std.DeleteFrames(frames=[0, 1]) + core.std.BlankClip(
            clip_down, length=2
        )
        shift_up1 = clip_down.std.DeleteFrames(frames=[0]) + core.std.BlankClip(
            clip_down, length=1
        )

        shift_down1 = core.std.BlankClip(clip_down, length=1) + core.std.BlankClip(
            clip_down, length=1
        )
        shift_down2 = core.std.BlankClip(clip_down, length=2) + core.std.BlankClip(
            clip_down, length=2
        )

        return core.std.ModifyFrame(
            clip,
            (clip, shift_down2, shift_down1, clip_down, shift_up1, shift_up2),
            execute,
        )

    elif onnx_type == "2img":
        clip_down = clip.resize.Bicubic(
            resolution[0], resolution[1], format=vs.RGBH, matrix_in_s="709"
        )

        return core.std.ModifyFrame(clip, (clip, clip_down, clip_down[1:]), execute)
