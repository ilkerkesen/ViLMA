from math import ceil
import click
import torch
from torchvision.io import read_video, write_png
from torchvision.transforms import Resize, CenterCrop
from vl_bench.utils import process_path


@click.command()
@click.option(
    '--video-file',
    required=True,
    type=click.Path(exists=True, file_okay=True),
)
@click.option(
    '--output-file',
    required=True,
    type=click.Path(),
)
@click.option(
    '--num-frames',
    type=int,
    default=5,
)
@click.option(
    '--size',
    type=int,
    default=256,
)
def main(video_file, output_file, num_frames, size):
    video_file = process_path(video_file)
    output_file = process_path(output_file)
    video_dict = read_video(video_file)
    resize, crop = Resize(size), CenterCrop(size)
    video_array = video_dict[0].permute(0, 3, 1, 2)
    T, C, H, W = video_array.shape
    step_size = ceil(T / num_frames)
    video_array = video_array[0:-1:step_size]
    video_array = crop(resize(video_array))
    video_summary = torch.cat([f for f in video_array], dim=2)
    write_png(video_summary, output_file)


if __name__ == "__main__":
    main()