import wget
import os
import sys
import tarfile

path = "/workspace/tensorrt/models/"
base_url = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def check_and_download(file_path: str):
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(file_path):
        model_name = os.path.basename(file_path)
        url = (
            base_url
            + model_name
        )
        eprint("downloading: " + model_name)
        wget.download(url, out=path)

def check_and_download_film():
    model_paths = []
    for model_type in ["L1", "Style", "VGG"]:
        model_paths.append(os.path.join(path, "FILM", model_type, "saved_model.pb"))
        model_paths.append(os.path.join(path, "FILM", model_type, "keras_metadata.pb"))
        model_paths.append(os.path.join(path, "FILM", model_type, "variables.data-00000-of-00001"))
        model_paths.append(os.path.join(path, "FILM", model_type, "variables.index"))
    
    for i in model_paths:
        if not os.path.exists(path):
            eprint("Models for 'FILM' not found. Downloading, please wait.")

            film_path = os.path.join(path, "FILM")
            if os.path.exists(film_path):
                os.remove(film_path)
            wget.download("https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/FILM.tar.gz")

            tar = tarfile.open("FILM.tar.gz")
            tar.extractall("/workspace/tensorrt/models/FILM/")
            tar.close()
            eprint("Download finished")
            break