import os
import os.path as osp
import json
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from .utils import process_path


class BaseDataset(Dataset):
    """
    Only loads the JSON annotations.
    """
    def __init__(self, json_path):
        json_path = process_path(json_path)
        assert osp.isfile(json_path)
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.ids = list(self.json_data.keys())
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_id = self.ids[index]
        return self.json_data[item_id]


class Dataset_v1(Dataset):
    """
    Read also the videos in addition to the raw JSON data.
    """
    def __init__(
        self,
        json_path,
        tokenizer=None,
        fps=None,
        num_frames=None,
        youtube_dir=None,
        quva_dir=None,
        **kwargs,
    ):
        self.json_data = BaseDataset(json_path)
        self.tokenizer = tokenizer
        self.fps = fps
        self.num_frames = num_frames
        self.kwargs = kwargs
        
        self.youtube_dir = None
        if youtube_dir is not None:
            self.youtube_dir = process_path(youtube_dir)
            
        self.quva_dir = None
        if quva_dir is not None:
            self.quva_dir = process_path(quva_dir)


    def _read_video(self, item):
        # find the full path
        dataset = item['dataset']
        video_file = item['video_file']
        if dataset == 'QUVA':
            video_dir = osp.join(self.quva_dir, 'videos')
        else:
            raise NotImplementedError('Not implemented yet.')
        video_path = osp.join(video_dir, video_file)

        if item['time_unit'] == 'sec':
            video, audio, fps = read_video(
                video_path,
                start_pts=float(item['start_time']),
                end_pts=float(item['end_time']),
                pts_unit='sec',
            )
        elif item['time_unit'] == 'pts':  # otherwise it returns single frame
            video, audio, fps = read_video(video_path)
            video = video[item['start_time']:item['end_time']]
        return video, audio, fps
        
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        entry = deepcopy(self.json_data[index])
        video, audio, fps = self._read_video(entry)
        raw_texts = [entry['caption']] + entry['foils']
        item = {
            'index': index,
            'item_id': self.json_data.ids[index],
            'video': video,
            'audio': audio,
            'video_fps': fps.get('video_fps'),
            'audio_fps': fps.get('audio_fps'),
            'raw_texts': raw_texts,
        }
        return item