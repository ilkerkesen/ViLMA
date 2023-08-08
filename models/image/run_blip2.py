import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import click
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from vl_bench.data import Dataset_v1
from vl_bench.utils import process_path


MODELS = (
    'Salesforce/blip2-opt-6.7b',
    'Salesforce/blip2-opt-2.7b',
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

    # read data
    data = Dataset_v1(
        input_file,
        quva_dir=quva_dir,
        something_something_dir=something_something_dir,
        youtube_dir=youtube_dir,
        proficiency=proficiency,
    )

    # initialize model & processor
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name).half().to(device)
    offset = model.config.num_query_tokens
    crit = nn.CrossEntropyLoss(
        reduction='none',
        ignore_index=processor.tokenizer.pad_token_id,
    )
    results = dict()
    for item in tqdm(data):
        video_len = item['video'].shape[0]
        clip_len = 8  # FIXME: hardcoded
        downsampled = item['video']
        if video_len > clip_len:
            indices = sample_frame_indices(
                clip_len=clip_len,
                frame_sample_rate=1,
                seg_len=item['video'].shape[0],
            )
            downsampled = item['video'][indices]

        inputs = processor(
            text=item['raw_texts'],
            images=list(downsampled),
            return_tensors='pt',
            padding=True,
        ).to(device)

        # reshape the arrays -- its different from CLIP/X-CLIP.
        num_texts = len(item['raw_texts'])
        num_frames = len(downsampled)
        inputs['pixel_values'] = inputs['pixel_values'].half()
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(1).repeat_interleave(num_texts, dim=1)
        C, H, W = inputs['pixel_values'].shape[2:]
        inputs['pixel_values'] = inputs['pixel_values'].reshape(-1, C, H, W)
        inputs['input_ids'] = inputs['input_ids'].unsqueeze(0).repeat_interleave(num_frames, dim=0)
        inputs['input_ids'] = inputs['input_ids'].reshape(-1, inputs['input_ids'].shape[-1])
        inputs['attention_mask'] = inputs['attention_mask'].unsqueeze(0).repeat_interleave(num_frames, dim=0)
        inputs['attention_mask'] = inputs['attention_mask'].reshape(-1, inputs['attention_mask'].shape[-1])

        with torch.no_grad():
            output = model(**inputs)

        logits = output.logits[:, offset:-1, :]
        labels = inputs['input_ids'][:, 1:].contiguous()
        lengths = inputs['attention_mask'].sum(dim=-1)
        scores = crit(logits.reshape(-1, logits.shape[-1]), labels.view(-1))
        scores = scores.reshape_as(labels)
        scores = scores.sum(dim=1) / lengths
        scores = scores.reshape(num_frames, num_texts)
        scores = scores.mean(dim=0).exp().tolist()
        item_id = item['item_id']
        results[item_id] = {'scores': scores}

    with open(process_path(output_file), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
