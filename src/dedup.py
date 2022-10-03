import cv2
from tqdm import tqdm
import numpy as np
import math
import torch
import numba
from numba import prange
import os 

# forcing printing by printing to error, normal print gets piped
# https://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


@numba.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False, nogil=True)
def PSNR(original, compressed):
    mse = 0.0
    for i in prange(original.shape[0]):
        for j in range(original.shape[1]):
            for k in range(3):
                mse += (original[i][j][k] - compressed[i][j][k]) ** 2
    mse = mse / (original.shape[0] * original.shape[1])
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def return_frames(filepath, psnr_value=35):
    cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames_duplicated = []

    appeared = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    eprint("Looking for duplicated frames")

    detected = False
    i = 0
    with tqdm(total=total_frames) as pbar:
        while i < total_frames:
            if detected == False:
                ret, img0 = cap.read()
                i += 1
                pbar.update(1)
                detected = True
            detected = False
            ret, img1 = cap.read()
            i += 1
            pbar.update(1)
            if PSNR(img0, img1) > psnr_value:
                frames_duplicated.append(i)
                detected = True

    original_frames = []
    duplicate_amount_frames = []
    for i in ranges(frames_duplicated):
        original_frames.append(i[0] - 1)
        duplicate_amount_frames.append(i[1] - i[0] + 1)

    frames_duplicating = []
    cumsum = np.cumsum(np.array(duplicate_amount_frames))
    cumsum = np.insert(cumsum, 0, 0)
    for i, j in enumerate(original_frames):
        for k in range(duplicate_amount_frames[i]):
            frames_duplicating.append(j - cumsum[i])

    eprint("\n")
    eprint("Frames duplicated: ", len(frames_duplicated))
    eprint("Duplicate Frames in %: ", (len(frames_duplicated) / total_frames) * 100)
    eprint("\n")

    return frames_duplicated, frames_duplicating


def get_duplicate_frames_with_vmaf(file_path: str):
    tmp_dir = "tmp/"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    txt_path = os.path.join(tmp_dir, "tmp.txt")

    if os.path.exists(txt_path):
        os.remove(txt_path)

    f_txt = open(txt_path, "w")
    f_txt.write(str(file_path))
    f_txt.close()

    os.system("vspipe parse.py -p .")

    ssimt = 0.999
    pxdifft = 10240
    consecutivet = 2

    with open(os.path.join("infos_running.txt"), "r") as f:
        lines = [i.split("\t") for i in f][1:]
    for i in range(len(lines)):
        lines[i][0] = int(lines[i][0])
        lines[i][1] = int(float(lines[i][1]) * 1000)
        lines[i][2] = float(lines[i][2])
        lines[i][3] = int(lines[i][3])
    lines.sort()
    startpts = lines[0][1]
    consecutive = 0

    dels = []
    tsv2o = []
    for i in range(len(lines)):
        l = lines[i]
        if l[2] >= ssimt and l[3] <= pxdifft and consecutive < consecutivet:
            consecutive += 1
            dels.append(l[0])
        else:
            consecutive = 0
            tsv2o.append(l[1] - startpts)
    return dels