# https://pyscenedetect.readthedocs.io/en/latest/reference/python-api/
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    result = scene_manager.get_scene_list()

    framelist = []
    from tqdm import tqdm
    for i in tqdm(range(len(result))):
        framelist.append(result[i][1].get_frames()*2-1)
        #print(result[i][1].get_frames())
        #print(result[i][1].get_timecode())

    return framelist
