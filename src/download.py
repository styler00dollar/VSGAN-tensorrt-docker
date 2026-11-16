import os
import sys
import urllib.parse

import requests
from tqdm import tqdm

path = "/workspace/tensorrt/models/"
base_url = (
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"
)


def check_and_download(file_path: str, chunk_size: int = 1 << 20, timeout: float = 60) -> None:
    os.makedirs(path, exist_ok=True)

    if os.path.exists(file_path):
        return

    model_name = os.path.basename(file_path)
    url = base_url + urllib.parse.quote(model_name)
    dest = os.path.join(path, model_name)
    tmp = dest + ".part"

    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0) or None
            with open(tmp, "wb") as f, tqdm(
                desc=model_name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                file=sys.stderr,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, dest)
    except (requests.RequestException, OSError) as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
