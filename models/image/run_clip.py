import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
from transformers import AutoProcessor, AutoModel
from vl_bench.data import Dataset_v2, get_clip_collate_fn
from vl_bench.utils import process_path


MODELS = (
    'openai/clip-vit-large-patch14',
    'openai/clip-vit-base-patch32',
)


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


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True
)
@click.option(
    '-m', '--model-name',
    type=click.Choice(choices=MODELS),
    default=MODELS[0],
)
@click.option(
    '--batch-size',
    type=int,
    default=16,
)
@click.option(
    '--num-workers',
    type=int,
    default=5,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '--quva-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '--something-something-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '--youtube-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--proficiency',
    is_flag=True,
)
def main(
    input_file,
    model_name,
    batch_size,
    num_workers,
    device,
    quva_dir,
    something_something_dir,
    youtube_dir,
    output_file,
    proficiency,
):
    # check video datasets' dirs
    assert quva_dir is not None \
        or something_something_dir is not None \
        or youtube_dir is not None
    if quva_dir is not None:
        quva_dir = process_path(quva_dir)
    if something_something_dir is not None:
        something_something_dir = process_path(something_something_dir)
    if youtube_dir is not None:
        youtube_dir = process_path(youtube_dir)
    np.random.seed(0)

    # initialize model & processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).half().to(device)

    # read data
    data = Dataset_v2(
        input_file,
        quva_dir=quva_dir,
        something_something_dir=something_something_dir,
        youtube_dir=youtube_dir,
        proficiency=proficiency,
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_clip_collate_fn(processor=processor),
        num_workers=num_workers,
        pin_memory=False,
    )

    results = dict()
    for i, batch in enumerate(tqdm(loader)):
        inputs = batch['inputs'].to(device)
        num_batch_texts = batch['num_texts']
        num_batch_frames = batch['num_frames']
        batch_size = len(num_batch_texts)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image
        
        text_offset, frame_offset = 0, 0
        for i in range(batch_size):
            num_texts = num_batch_texts[i]
            num_frames = num_batch_frames[i]
            text_start = text_offset
            text_end = text_offset + num_texts
            frame_start = frame_offset
            frame_end = frame_offset + num_frames
            scores = logits[frame_start:frame_end, text_start:text_end]
            scores = scores.softmax(dim=1).mean(dim=0).cpu().tolist()
            key = batch['item_ids'][i]
            results[key] = {'scores': scores}
            text_offset += num_texts
            frame_offset += num_frames

    with open(process_path(output_file), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
