import os
import os.path as osp
import json
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from .utils import process_path


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len > converted_len:
        end_idx = np.random.randint(converted_len, seg_len)
    else:
        end_idx = min(converted_len, seg_len)-1
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


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
        for item_id in self.ids:
            self.json_data[item_id]['item_id'] = item_id

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
        something_something_dir=None,
        proficiency=False,
        **kwargs,
    ):
        self.json_data = BaseDataset(json_path)
        self.tokenizer = tokenizer
        self.fps = fps
        self.num_frames = num_frames
        self.kwargs = kwargs
        self.proficiency = proficiency

        self.youtube_dir = None
        if youtube_dir is not None:
            self.youtube_dir = process_path(youtube_dir)

        self.quva_dir = None
        if quva_dir is not None:
            self.quva_dir = process_path(quva_dir)

        self.something_something_dir = None
        if something_something_dir is not None:
            self.something_something_dir = process_path(something_something_dir)


    def _read_video(self, item):
        # find the full path
        dataset = item['dataset']
        video_file = item['video_file']
        video_path = None
        if dataset == 'QUVA':
            normalized = item.get('normalized')
            assert normalized
            video_dir = osp.join(self.quva_dir, 'normalized_videos')
            video_path = osp.join(video_dir, video_file)
        elif dataset == 'something-something-v2':
            video_dir = self.something_something_dir
            video_path = osp.join(video_dir, f'{item["dataset_idx"]}.webm')
        elif dataset == 'RareAct' or dataset == 'VidSitu':
            video_dir = self.youtube_dir
            video_path = osp.join(video_dir, f'{item["youtube_id"]}.mp4')
        else:
            raise NotImplementedError('Not implemented yet.')

        start_pts = item.get('start_time')
        end_pts = item.get('end_time', -1)
        end_pts = end_pts if end_pts != -1 else None

        if item['time_unit'] == 'sec':
            end_pts = float(end_pts) if end_pts is not None else None
            video = read_video(
                video_path,
                start_pts=float(start_pts),
                end_pts=end_pts,
                pts_unit='sec',
                output_format='TCHW',
            )[0]
        elif item['time_unit'] == 'pts':  # otherwise it returns single frame
            video = read_video(video_path, output_format='TCHW')[0]
            video = video[item['start_time']:item['end_time']]
        return video

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        entry = deepcopy(self.json_data[index])
        try:
            video = self._read_video(entry)
        except RuntimeError:
            video = None
        subentry = entry if not self.proficiency else entry['proficiency']
        raw_texts = [subentry['caption']] + subentry['foils']

        item = {
            'index': index,
            'item_id': self.json_data.ids[index],
            'video': video,
            'raw_texts': raw_texts,
        }
        return item


class Dataset_v2(Dataset_v1):
    def __init__(self, *args, clip_len=8, frame_sample_rate=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['num_texts'] = len(item['raw_texts'])

        downsampled = item.pop('video')
        video_len = downsampled.shape[0]
        if video_len > self.clip_len:
            indices = sample_frame_indices(
                clip_len=self.clip_len,
                frame_sample_rate=self.frame_sample_rate,
                seg_len=video_len,
            )
            downsampled = downsampled[indices]
        downsampled = list(downsampled)
        diff = self.clip_len - len(downsampled)
        if diff > 0:
            downsampled += diff * [downsampled[-1]]
        item['video'] = downsampled
        return item


def get_xclip_collate_fn(processor, dtype=torch.half):
    def _collate_fn(batch):
        item_ids = [x['item_id'] for x in batch]
        num_texts = [x['num_texts'] for x in batch]

        texts = []
        for item in batch:
            texts.extend(item['raw_texts'])

        inputs = processor(
            text=texts,
            videos=[x['video'] for x in batch],
            return_tensors='pt',
            padding=True,
        )
        if dtype == torch.half:
            inputs['pixel_values'] = inputs['pixel_values'].half()

        return {
            'inputs': inputs,
            'item_ids': item_ids,
            'num_texts': num_texts,
        }
    return _collate_fn


def get_clip_collate_fn(processor):
    def _collate_fn(batch):
        item_ids = [x['item_id'] for x in batch]
        num_texts = [x['num_texts'] for x in batch]
        num_frames = [len(x['video']) for x in batch]

        texts, frames = [], []
        for item in batch:
            texts.extend(item['raw_texts'])
            frames.extend(item['video'])

        inputs = processor(
            text=texts,
            images=frames,
            return_tensors='pt',
            padding=True,
        )
        inputs['pixel_values'] = inputs['pixel_values'].half()

        return {
            'inputs': inputs,
            'item_ids': item_ids,
            'num_texts': num_texts,
            'num_frames': num_frames,
        }
    return _collate_fn