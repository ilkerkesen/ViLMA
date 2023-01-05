import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import click
from transformers import AutoProcessor, AutoModel
from vl_bench.data import Dataset_v1
from vl_bench.utils import process_path


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


@click.command()
@click.option(
    '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True
)
@click.option(
    '--batch-size',
    type=int,
    default=1,
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
def main(input_file, batch_size, device, quva_dir, something_something_dir):
    # check video datasets' dirs
    assert quva_dir is not None or something_something_dir is not None
    if quva_dir is not None:
        quva_dir = process_path(quva_dir)
    if something_something_dir is not None:
        something_something_dir = process_path(something_something_dir)
    np.random.seed(0)

    # read data
    data = Dataset_v1(
        input_file,
        quva_dir=quva_dir,
        something_something_dir=something_something_dir,    
    )

    # initialize model & processor
    model_name = "microsoft/xclip-base-patch32"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    results = dict()
    for item in tqdm(data):
        indices = sample_frame_indices(  # FIXME: hardcoded
            clip_len=8,
            frame_sample_rate=1,
            seg_len=item['video'].shape[0],
        )
        downsampled = item['video'][indices]
        inputs = processor(
            text=item['raw_texts'],
            videos=list(downsampled),
            return_tensors='pt',
            padding=True,
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
        scores = output.logits_per_video.softmax(dim=-1).tolist()[0]
        item_id = item['item_id']
        results[item_id] = scores

    n_correct = 0
    for scores in results.values():
        caption_score = scores[0]
        foil_scores = scores[1:]
        caption_predicted = True
        for score in foil_scores:
            if score > caption_score:
                caption_predicted = False
        if caption_predicted:
            n_correct += 1
    accuracy = round(100 * n_correct / len(data), 2)
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
