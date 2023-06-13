import itertools
import numpy as np
import vapoursynth as vs
import functools
import torch
import torchvision.transforms as T
import cv2
import timm
import torchvision
from .download import check_and_download
import traceback
import sys
import onnxruntime as ort
from scipy.special import softmax
from threading import Lock


# todo: 
# - onnx only supports fp16, make workaround
# - threading with onnx
# - tensorrt
def scene_detect(
    clip: vs.VideoNode,
    model_name: str = "efficientnetv2b0+rife46",
    thresh: float = 0.98,
    fp16: bool = True,
    onnx: bool = False,
    onnx_path: str = "sc_efficientformerv2_s0+rife46_84119_224_fp16_op16.onnx",
    onnx_res: int = 224,
    num_streams: int = 4,
) -> vs.VideoNode:
    core = vs.core
    use_rife = False

    def frame_to_tensor(frame: vs.VideoFrame):
        return np.stack(
            [np.asarray(frame[plane]) for plane in range(frame.format.num_planes)]
        )

    def tensor_to_frame(f: vs.VideoFrame, array) -> vs.VideoFrame:
        for plane in range(f.format.num_planes):
            d = np.asarray(f[plane])
            np.copyto(d, array[plane, :, :])
        return f

    def tensor_to_clip(clip: vs.VideoNode, image) -> vs.VideoNode:
        clip = core.std.BlankClip(
            clip=clip, width=image.shape[-1], height=image.shape[-2]
        )
        return core.std.ModifyFrame(
            clip=clip,
            clips=clip,
            selector=lambda n, f: tensor_to_frame(f.copy(), image),
        )

    if not onnx:
        if model_name == "efficientnetv2_b0":
            model_path = "/workspace/tensorrt/models/sc_efficientnetv2b0_17957_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "tf_efficientnetv2_b0", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False

        elif model_name == "efficientnetv2b0+rife46":
            model_path = "/workspace/tensorrt/models/sc_efficientnetv2b0+rife46_flow_1362_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "tf_efficientnetv2_b0", num_classes=2, pretrained=False, in_chans=22
            )
            use_rife = True
            resolution = 256
            video_arch = False

        elif model_name == "efficientformerv2_s0":
            model_path = (
                "/workspace/tensorrt/models/sc_efficientformerv2_s0_12263_224.pth"
            )
            check_and_download(model_path)
            from src.sc.efficientformer_v2_arch import efficientformerv2_s0

            model = efficientformerv2_s0(in_ch=6)
            resolution = 224
            video_arch = False
        elif model_name == "efficientformerv2_s0+rife46":
            model_path = "/workspace/tensorrt/models/sc_efficientformerv2_s0+rife46_84119_224.pth"
            check_and_download(model_path)
            from src.sc.efficientformer_v2_arch import efficientformerv2_s0

            model = efficientformerv2_s0(in_ch=22)
            use_rife = True
            resolution = 224
            video_arch = False

        elif model_name == "maxvit_small":
            model_path = "/workspace/tensorrt/models/sc_maxvit_small_9072_224.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "maxvit_small_224", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 224
            video_arch = False
        elif model_name == "maxvit_small+rife46":
            model_path = (
                "/workspace/tensorrt/models/sc_maxvit_small+rife46_1512_224.pth"
            )
            check_and_download(model_path)
            model = timm.create_model(
                "maxvit_small_224", num_classes=2, pretrained=False, in_chans=22
            )
            use_rife = True
            resolution = 224
            video_arch = False
        elif model_name == "regnetz_005":
            model_path = "/workspace/tensorrt/models/sc_regnetz_005_33142_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "regnetz_005", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False
        elif model_name == "repvgg_b0":
            model_path = "/workspace/tensorrt/models/sc_repvgg_b0_7575_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "repvgg_b0", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False
        elif model_name == "resnetrs50":
            model_path = "/workspace/tensorrt/models/sc_resnetrs50_4840_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "resnetrs50", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False
        elif model_name == "resnetv2_50":
            model_path = "/workspace/tensorrt/models/sc_resnetv2_50_1815_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "resnetv2_50", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False
        elif model_name == "rexnet_100":
            model_path = "/workspace/tensorrt/models/sc_rexnet_100_7264_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "rexnet_100", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False
        elif model_name == "swinv2_small":
            model_path = (
                "/workspace/tensorrt/models/sc_swinv2_small_window16_10412_256.pth"
            )
            check_and_download(model_path)
            model = timm.create_model(
                "swinv2_small_window16_256", num_classes=2, pretrained=False, in_chans=6
            )
            resolution = 256
            video_arch = False
        elif model_name == "swinv2_small+rife46":
            model_path = "/workspace/tensorrt/models/sc_swinv2_small_window16+rife46_1814_256.pth"
            check_and_download(model_path)
            model = timm.create_model(
                "swinv2_small_window16_256",
                num_classes=2,
                pretrained=False,
                in_chans=22,
            )
            use_rife = True
            resolution = 256
            video_arch = False
        elif model_name == "TimeSformer":
            model_path = "/workspace/tensorrt/models/sc_TimeSformer_2592_224.pth"
            check_and_download(model_path)
            from timesformer_pytorch import TimeSformer

            model = TimeSformer(
                dim=512,
                image_size=224,
                patch_size=16,
                num_frames=2,
                num_classes=2,
                depth=12,
                heads=8,
                dim_head=64,
                attn_dropout=0.1,
                ff_dropout=0.1,
            )
            resolution = 224
            video_arch = True
        elif model_name == "uniformerv2_b16":
            model_path = "/workspace/tensorrt/models/sc_uniformerv2_b16_36288_224.pth"
            check_and_download(model_path)
            from src.sc.uniformerv2_arch import uniformerv2_b16

            model = uniformerv2_b16(
                pretrained=False,
                use_checkpoint=False,
                checkpoint_num=[0],
                t_size=2,
                dw_reduction=1.5,
                backbone_drop_path_rate=0.0,
                temporal_downsample=True,
                no_lmhra=False,
                double_lmhra=True,
                return_list=[8, 9, 10, 11],
                n_layers=4,
                n_dim=768,
                n_head=12,
                mlp_factor=4.0,
                drop_path_rate=0.0,
                mlp_dropout=[0.5, 0.5, 0.5, 0.5],
                cls_dropout=0.5,
                num_classes=2,
                frozen=False,
            )
            resolution = 224
            video_arch = True

        model.load_state_dict(torch.load(model_path))
        model.cuda().eval()

        if fp16:
            model.half()

        if use_rife:
            from .rife_arch import IFNet

            model_path = "/workspace/tensorrt/models/rife46.pth"
            check_and_download(model_path)
            rife_model = IFNet(arch_ver="4.6")
            rife_model.load_state_dict(torch.load(model_path), True)
            rife_model.eval().cuda()

            if fp16:
                rife_model.half()

    if onnx:
        model_path = "/workspace/tensorrt/models/sc_efficientformerv2_s0+rife46_84119_224_fp16_op16.onnx"
        check_and_download(model_path)
        sess = ort.InferenceSession(
            model_path,
            providers=["TensorrtExecutionProvider"],
        )
        resolution = onnx_res

    stream = [torch.cuda.Stream(device="cuda") for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]
    index = -1
    index_lock = Lock()
    torch._dynamo.config.suppress_errors = True

    @torch.compile()
    @torch.inference_mode()
    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        if not onnx:
            with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
                I0 = frame_to_tensor(clip.get_frame(n))
                I1 = frame_to_tensor(clip.get_frame(n + 1))
                I0 = np.rollaxis(I0, 0, 3)
                I1 = np.rollaxis(I1, 0, 3)
                I0 = torch.from_numpy(I0).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                I1 = torch.from_numpy(I1).unsqueeze(0).permute(0, 3, 1, 2).cuda()

                I0 = torch.nn.functional.interpolate(I0, size=(resolution, resolution), mode="area")
                I1 = torch.nn.functional.interpolate(I1, size=(resolution, resolution), mode="area")

                with torch.inference_mode():
                    if not video_arch:
                        img = torch.cat([I0, I1], dim=1)
                    elif model_name == "TimeSformer":
                        img = torch.stack([I0, I1], dim=0)
                    elif model_name == "uniformerv2_b16":
                        img = (
                            torch.stack([I0.squeeze(0), I1.squeeze(0)], dim=0)
                            .unsqueeze(0)
                            .permute(0, 2, 1, 3, 4)
                        )
                    if use_rife:
                        if fp16:
                            out = rife_model(I0.half(), I1.half(), return_flow=True)
                        if not fp16:
                            out = rife_model(I0, I1, return_flow=True)
                        for i in range(4):
                            img = torch.cat(
                                [
                                    img.cuda(),
                                    out[i][:, :, :resolution, :resolution].cuda(),
                                ],
                                dim=1,
                            )

                    if fp16:
                        img = img.half()

                    result = model(img)
                    y_prob = torch.softmax(result, dim=1)
                    if y_prob[0][0] > thresh:
                        return core.std.SetFrameProp(
                            clip, prop="_SceneChangeNext", intval=1
                        )
                    return clip
        if onnx:
            I0 = frame_to_tensor(clip.get_frame(n))
            I1 = frame_to_tensor(clip.get_frame(n + 1))
            I0 = np.rollaxis(I0, 0, 3)
            I1 = np.rollaxis(I1, 0, 3)
            I0 = I0.astype(np.float32)
            I1 = I1.astype(np.float32)
            I0 = cv2.resize(I0, (resolution, resolution), interpolation=cv2.INTER_AREA)
            I1 = cv2.resize(I1, (resolution, resolution), interpolation=cv2.INTER_AREA)
            I0 = np.expand_dims(I0, 0)
            I0 = np.swapaxes(I0, 3, 1)
            I0 = np.swapaxes(I0, 2, 3)

            I1 = np.expand_dims(I1, 0)
            I1 = np.swapaxes(I1, 3, 1)
            I1 = np.swapaxes(I1, 2, 3)

            input = np.concatenate([I0, I1], axis=1)
            input = input.astype(np.float16)
            result = sess.run(None, {"input": input})[0]

            y_prob = softmax(result, axis=1)
            if y_prob[0][0] > thresh:
                return core.std.SetFrameProp(clip, prop="_SceneChangeNext", intval=1)
        return clip

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )
