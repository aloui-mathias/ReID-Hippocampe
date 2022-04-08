# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""


def notebook_init(verbose=True):
    # Check system software and hardware

    import os
    import shutil

    from .general import check_requirements, emojis, is_colab
    from .torch_utils import select_device  # imports

    check_requirements(('psutil', 'IPython'))
    import psutil
    from IPython import display  # to display images and clear console output

    if is_colab():
        # remove colab /sample_data directory
        shutil.rmtree('/content/sample_data', ignore_errors=True)

    if verbose:
        # System info
        # gb = 1 / 1000 ** 3  # bytes to GB
        gib = 1 / 1024 ** 3  # bytes to GiB
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram * gib:.1f} GB RAM, {(total - free) * gib:.1f}/{total * gib:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False)
    return display
