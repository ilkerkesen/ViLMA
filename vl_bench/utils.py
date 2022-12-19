import os.path as osp


def process_path(path):
    path = osp.expanduser(path)
    path = osp.abspath(path)
    return path
