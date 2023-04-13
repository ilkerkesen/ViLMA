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
        with open(json_path, "r") as f:
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
        youcook2_dir=None,
        rareact_dir=None,
        coin_dir=None,
        star_dir=None,
        something_something_dir=None,
        cache_dir=None,
        **kwargs,
    ):
        self.json_data = BaseDataset(json_path)
        self.tokenizer = tokenizer
        self.fps = fps
        self.num_frames = num_frames
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        self.youtube_dir = None
        if youtube_dir is not None:
            self.youtube_dir = process_path(youtube_dir)

        self.quva_dir = None
        if quva_dir is not None:
            self.quva_dir = process_path(quva_dir)

        self.something_something_dir = None
        if something_something_dir is not None:
            self.something_something_dir = process_path(something_something_dir)

        self.youcook2_dir = None
        if youcook2_dir is not None:
            self.youcook2_dir = process_path(youcook2_dir)

        self.rareact_dir = None
        if rareact_dir is not None:
            self.rareact_dir = process_path(rareact_dir)

        self.coin_dir = None
        if coin_dir is not None:
            self.coin_dir = process_path(coin_dir)

        self.star_dir = None
        if star_dir is not None:
            self.star_dir = process_path(star_dir)

    def _read_video(self, item, index):
        # find the full path
        dataset = item["dataset"]
        video_path = None
        if dataset == "QUVA":
            video_file = item["video_file"]
            normalized = item.get("normalized")
            assert normalized
            video_dir = osp.join(self.quva_dir, "normalized_videos")
            video_path = osp.join(video_dir, video_file)  # TODO: extension?
        elif dataset == "something-something-v2":
            video_file = item["video_file"]
            video_dir = self.something_something_dir
            video_path = osp.join(video_dir, f"{video_file}.webm")
        elif dataset == "youcook2":
            video_file = item["youtube_id"]
            video_dir = self.youcook2_dir
            video_path = osp.join(video_dir, f"{video_file}.mp4")
        elif dataset == "rareact":
            video_file = item["youtube_id"]
            video_dir = self.rareact_dir
            video_path = osp.join(video_dir, f"{video_file}.mp4")
        elif dataset == "coin":
            video_file = item["youtube_id"]
            video_dir = self.coin_dir
            video_path = osp.join(video_dir, f"{video_file}.mp4")
        elif dataset == "star":
            video_file = item["video_file"]
            video_dir = self.star_dir
            video_path = osp.join(video_dir, f"{video_file}.mp4")
        else:
            raise NotImplementedError("Not implemented yet.")

        start_pts = item.get("start_time")
        end_pts = item.get("end_time", -1)
        end_pts = end_pts if end_pts != -1 else None
        start_pts = start_pts if start_pts is not None else 0

        cached_file = (
            os.path.join(self.cache_dir, str(self.json_data.ids[index])) + ".mp4"
        )
        if os.path.exists(cached_file):
            return None, None, cached_file

        if item["time_unit"] == "sec":
            end_pts = float(end_pts) if end_pts is not None else None
            video = read_video(
                video_path,
                start_pts=float(start_pts),
                end_pts=end_pts,
                pts_unit="sec",
                output_format="TCHW",
            )
            fps = video[-1]["video_fps"]
            video = video[0]
        elif item["time_unit"] == "pts":  # otherwise it returns single frame
            video = read_video(video_path, output_format="TCHW")
            fps = video[-1]["video_fps"]
            video = video[0]
            video = video[item["start_time"] : item["end_time"]]

        return video, fps, video_path

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        entry = deepcopy(self.json_data[index])
        video, fps, video_path = self._read_video(entry, index)
        raw_texts = [entry["caption"]] + entry["foils"]
        item = {
            "index": index,
            "item_id": self.json_data.ids[index],
            "video": video,
            "raw_texts": raw_texts,
            "fps": fps,
            "start_time": entry["start_time"],
            "end_time": entry["end_time"],
            "video_path": video_path,
        }
        return item
