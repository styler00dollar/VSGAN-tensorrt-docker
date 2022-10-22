import wget
import os
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def check_and_download(file_path: str):
    path = "models/"
    if not os.path.exists(path):
        os.mkdir(tmp_dir)

    if not os.path.exists(file_path)
        model_name = os.path.basename(file_path)
        url = (
            "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
            + model_name
        )
        eprint("downloading: " + model_name)
        wget.download(url, out=path)
