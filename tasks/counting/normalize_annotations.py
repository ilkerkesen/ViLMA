import os
import os.path as osp
import numpy as np
import click
from tqdm import tqdm
from torchvision.io import read_video
from vl_bench.utils import process_path


@click.command()
@click.option(
    '--data-dir',
    type=click.Path(dir_okay=True, exists=True),
    required=True,
)
@click.option(
    '--target-fps',
    type=int,
    default=30,
)
def main(data_dir, target_fps):
    data_dir = process_path(data_dir)
    save_dir = osp.join(data_dir, 'normalized_annotations')
    annotations_dir = osp.join(data_dir, 'annotations')
    videos_dir = osp.join(data_dir, 'videos')

    if not osp.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    prefixes = [osp.splitext(x)[0] for x in os.listdir(annotations_dir)]
    for prefix in tqdm(prefixes):
        video_path = osp.join(videos_dir, f'{prefix}.mp4')
        annotation_path = osp.join(annotations_dir, f'{prefix}.npy')
        video_item = read_video(video_path)
        num_frames = video_item[0].shape[0]
        fps = video_item[2]['video_fps']
        raw_ann = np.load(annotation_path)
        normalized = raw_ann / fps * target_fps
        normalized = normalized.round().astype(int)
        save_path = osp.join(save_dir, f'{prefix}.npy')
        np.save(save_path, normalized)


if __name__ == "__main__":
    main()