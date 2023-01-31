import vapoursynth as vs
import color_matcher
from color_matcher.normalizer import Normalizer
import numpy as np


def vs_color_match(clip, reference_clip, method=None):
    """

    :param clip: An RGB24 color clip, what will be transformed
    :param reference_clip: An RGB24 color clip, the clip to use as the color reference
    :param method: The transfer algorithm
    :return: VS clip
    """

    cm = color_matcher.ColorMatcher()

    def do_color_transfer(n, f):
        fout = f[1].copy()
        before_cm = frame_to_cm(f[0])
        after_cm = frame_to_cm(f[1])
        fixed = cm.transfer(after_cm, ref=before_cm, method=method)
        cm_to_frame(fixed, fout)
        return fout

    return vs.core.std.ModifyFrame(clip=clip, clips=[reference_clip, clip], selector=do_color_transfer)


# Adapted from color-matcher's io handlers

def cm_to_frame(img: np.ndarray, frame):
    int_img = Normalizer(img).uint8_norm()
    for p in range(3):
        pls = frame[p]
        frame_arr = np.asarray(pls)
        slice = int_img[:, :, p]
        np.copyto(frame_arr, int_img[:, :, p])


def frame_to_cm(frame: str = None) -> np.ndarray:
    arr = np.dstack([
        np.asarray(frame[p])
        for p in [0, 1, 2]
    ])
    return Normalizer(arr).type_norm()
