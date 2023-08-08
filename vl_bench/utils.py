import os.path as osp
import torch


def process_path(path):
    path = osp.expanduser(path)
    path = osp.abspath(path)
    return path


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def remove_count_phrase(template):
    template = template.replace(' in exactly <number> moves', '')
    template = template.replace(' exactly <number> times', '')
    template = template.replace(' exactly <number>', '')
    template = template.replace(' <number> times in total', '')
    return template