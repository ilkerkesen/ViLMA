import os
import os.path as osp
import json
import click
from vl_bench.utils import process_path


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(exists=False, file_okay=True),
    required=True,
)
@click.option(
    '--video-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
def main(input_file, output_file, video_dir):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    video_dir = process_path(video_dir)
    video_list = os.listdir(video_dir)
    youtube_ids = [osp.splitext(x)[0] for x in video_list]
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    output_data = dict()
    for key, item in input_data.items():
        if item['youtube_id'] in youtube_ids:
            output_data[key] = item
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()