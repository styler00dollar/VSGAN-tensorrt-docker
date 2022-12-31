import os
import numpy as np

ssimt = 0.999
pxdifft = 10240
consecutivet = 2
tmp_dir = "tmp/"

# https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# frames to delete
def processInfo():
    with open(os.path.join(tmp_dir, "infos_running.txt"), "r") as f:
        lines = [i.split("\t") for i in f][1:]
    for i in range(len(lines)):
        lines[i][0] = int(lines[i][0])
        lines[i][1] = int(float(lines[i][1]) * 1000)
        lines[i][2] = float(lines[i][2])
    lines.sort()
    startpts = lines[0][1]
    consecutive = 0

    dels = []
    for i in range(len(lines)):
        l = lines[i]
        if l[2] >= ssimt and consecutive < consecutivet:
            consecutive += 1
            dels.append(l[0])
        else:
            consecutive = 0
    return dels


def get_dedup_frames():
    frames_duplicated = processInfo()

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

    return frames_duplicated, frames_duplicating