from math import ceil
import click
import torch
from torchvision.io import read_video, write_png
from torchvision.transforms import Resize, CenterCrop
from vl_bench.utils import process_path


@click.command()
@click.option(
    '-i', '--video-file',
    required=True,
    type=click.Path(exists=True, file_okay=True),
)
@click.option(
    '-o', '--output-file',
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
@click.option(
    '-s', '--start-pts',
    type=float,
    default=0.0,
)
@click.option(
    '-e', '--end-pts',
    type=float,
    default=None,
)
@click.option(
    '--pts-unit',
    type=click.Choice(choices=['pts', 'sec']),
    default='sec',
)
def main(video_file, output_file, num_frames, size, start_pts, end_pts, pts_unit):
    video_file = process_path(video_file)
    output_file = process_path(output_file)
    if pts_unit == 'sec':
        video_dict = read_video(
            video_file,
            start_pts=float(start_pts),
            end_pts=float(end_pts) if end_pts is not None else None,
            pts_unit=pts_unit,
        )
        video_array = video_dict[0]
    elif pts_unit == 'pts':
        video_dict = read_video(
            video_file,
            # start_pts=int(start_pts),
            # end_pts=int(end_pts),
            pts_unit=pts_unit,
        )
        video_array = video_dict[0][int(start_pts):int(end_pts)]
    resize, crop = Resize(size), CenterCrop(size)
    video_array = video_array.permute(0, 3, 1, 2)
    T, C, H, W = video_array.shape
    step_size = ceil(T / num_frames)
    video_array = video_array[0:-1:step_size]
    video_array = crop(resize(video_array))
    video_summary = torch.cat([f for f in video_array], dim=2)
    write_png(video_summary, output_file)


if __name__ == "__main__":
    main()