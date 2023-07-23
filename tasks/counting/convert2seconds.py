from copy import deepcopy
import os
import os.path as osp
import json
import click
from torchvision.io import read_video_timestamps
from tqdm import tqdm
from vl_bench.utils import process_path


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '-d', '--data-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    '--fps',
    type=float,
    default=30.0,
    show_default=True,
)
def main(input_file, output_file, data_dir, fps):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    data_dir = process_path(data_dir)
    video_dir = osp.join(data_dir, 'normalized_videos')

    with open(input_file, 'r') as f:
        data = json.load(f)

    processed = dict()
    for item_id, item in tqdm(data.items()):
        processed[item_id] = deepcopy(item)
        start_time = round(item['start_time'] / fps, 4)  # FIXME: hardcoded.
        end_time = round(item['end_time'] / fps, 4)
        processed[item_id]['start_time'] = start_time
        processed[item_id]['end_time'] = end_time
        processed[item_id]['time_unit'] = 'sec'
    
    with open(output_file, 'w') as f:
        json.dump(processed, f, sort_keys=False, indent=4)
        

if __name__ == "__main__":
    main()