# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

import subprocess
import urllib
from pathlib import Path

import requests


def gsutil_getsize(url=''):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f'gsutil du {url}', shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # bytes


def attempt_download(file, repo='ultralytics/yolov5'):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        # URL specified
        # decode '%2F' to '/' etc.
        name = Path(urllib.parse.unquote(str(file))).name
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            # parse authentication https://url.com/file.txt?auth...
            file = name.split('?')[0]
            return file

        # GitHub assets
        # make parent dir (if required)
        file.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(
                f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']  # i.e. 'v1.0'
        except Exception:  # fallback plan
            assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output(
                    'git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except Exception:
                tag = 'v6.0'  # current release

        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)
