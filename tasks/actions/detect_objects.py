import json
import numpy as np
import click
from tqdm import tqdm
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from vl_bench.utils import process_path
from vl_bench.data import Dataset_v1


SUPPORTED_MODELS = (
    'facebook/detr-resnet-50',
    'facebook/detr-resnet-101',
    'facebook/detr-resnet-50-dc5',
    'facebook/detr-resnet-101-dc5',
)

# sample indices -- same with the X-CLIP.
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


def post_process_predictions(model, processor, outputs, threshold):
    results = processor.post_process_object_detection(
        outputs, threshold=threshold)
    labels = torch.cat([x['labels'] for x in results], dim=-1)
    labels = sorted(list(set(labels.tolist())))
    id2label = model.config.id2label
    return [id2label[x] for x in labels if x != 0]


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
@click.option(
    '-m', '--model-name',
    type=click.Choice(choices=SUPPORTED_MODELS),
    default=SUPPORTED_MODELS[0],
    show_default=True,
)
@click.option(
    '--device',
    type=click.Choice(choices=['cuda:0', 'cpu']),
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '-T', '--threshold',
    type=float,
    default=0.8,
    show_default=True,
)
def main(
    input_file,
    output_file,
    video_dir,
    model_name,
    device,
    threshold,
):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    video_dir = process_path(video_dir)
    device = torch.device(device)
    dtype = torch.float16  # FIXME: hardcoded.
    
    # load data
    input_data = Dataset_v1(
        json_path=input_file,
        tokenizer=None,
        youtube_dir=video_dir,
    )
    output_data = dict()

    # load model & processor
    model = DetrForObjectDetection.from_pretrained(model_name)
    model = model.to(device, dtype)
    model.eval()
    processor = DetrImageProcessor.from_pretrained(model_name)

    for i, item in enumerate(tqdm(input_data)):
        item_id = item['item_id']
        if item['video'] is None:
            output_data[item_id] = list()
            continue

        indices = sample_frame_indices(  # FIXME: hardcoded
            clip_len=8,
            frame_sample_rate=1,
            seg_len=item['video'].shape[0],
        )
        downsampled = item['video'][indices]
        inputs = processor(
            images=downsampled,
            return_tensors='pt',
        ).to(device, dtype)
        with torch.no_grad():
            outputs = model(**inputs)
        detected = post_process_predictions(
            model=model,
            processor=processor,
            outputs=outputs,
            threshold=threshold,
        )
        output_data[item_id] = detected

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()