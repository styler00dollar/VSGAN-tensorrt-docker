# https://pyscenedetect.readthedocs.io/en/latest/reference/python-api/
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
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


def find_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    result = scene_manager.get_scene_list()

    framelist = []
    from tqdm import tqdm

    for i in tqdm(range(len(result))):
        framelist.append(result[i][1].get_frames() * 2 - 1)
        # print(result[i][1].get_frames())
        # print(result[i][1].get_timecode())

    return framelist


def scene_detect(
    clip: vs.VideoNode,
    model_name: str = "efficientnetv2_b0",
    thresh: float = 0.98,
    fp16: bool = False,
) -> vs.VideoNode:
    core = vs.core

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

    if model_name == "efficientnetv2_b0":
        model_path = "/workspace/tensorrt/models/sc_efficientnetv2b0_17957_256.pth"
        check_and_download(model_path)
        model = timm.create_model(
            "tf_efficientnetv2_b0", num_classes=2, pretrained=False, in_chans=6
        )
        resolution = 256
        video_arch = False
    elif model_name == "efficientformerv2_s0":
        model_path = "/workspace/tensorrt/models/sc_efficientformerv2_s0_12263_224.pth"
        check_and_download(model_path)
        from src.sc.efficientformer_v2_arch import efficientformerv2_s0

        model = efficientformerv2_s0()
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
        model_path = "/workspace/tensorrt/models/sc_swinv2_small_window16_10412_256.pth"
        check_and_download(model_path)
        model = timm.create_model(
            "swinv2_small_window16_256", num_classes=2, pretrained=False, in_chans=6
        )
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

    def execute(n: int, clip: vs.VideoNode) -> vs.VideoNode:
        I0 = frame_to_tensor(clip.get_frame(n))
        I1 = frame_to_tensor(clip.get_frame(n + 1))
        I0 = np.rollaxis(I0, 0, 3)
        I1 = np.rollaxis(I1, 0, 3)
        I0 = cv2.resize(I0, (resolution, resolution), interpolation=cv2.INTER_AREA)
        I1 = cv2.resize(I1, (resolution, resolution), interpolation=cv2.INTER_AREA)
        I0 = torch.from_numpy(I0).unsqueeze(0).permute(0, 3, 1, 2)
        I1 = torch.from_numpy(I1).unsqueeze(0).permute(0, 3, 1, 2)

        with torch.inference_mode():
            if not video_arch:
                img = torch.cat([I0, I1], dim=1).cuda()
            elif model_name == "TimeSformer":
                img = torch.stack([I0, I1], dim=0).cuda()
            elif model_name == "uniformerv2_b16":
                img = (
                    torch.stack([I0.squeeze(0), I1.squeeze(0)], dim=0)
                    .cuda()
                    .unsqueeze(0)
                    .permute(0, 2, 1, 3, 4)
                )
            if fp16:
                img = img.half()
            result = model(img)

        y_prob = torch.softmax(result, dim=1)
        if y_prob[0][0] > thresh:
            return core.std.SetFrameProp(clip, prop="_SceneChangeNext", intval=1)
        return clip

    return core.std.FrameEval(
        core.std.BlankClip(clip=clip, width=clip.width, height=clip.height),
        functools.partial(execute, clip=clip),
    )
