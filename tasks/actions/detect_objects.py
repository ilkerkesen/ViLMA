import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import click
from tqdm import tqdm
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.io import read_video as _read_video
from vl_bench.utils import process_path



SUPPORTED_MODELS = (
    'facebook/detr-resnet-50',
    'facebook/detr-resnet-101',
    'facebook/detr-resnet-50-dc5',
    'facebook/detr-resnet-101-dc5',
)


def read_video(video_dir, video_id, start_pts, end_pts):
    return _read_video(
        osp.join(video_dir, f'{video_id}.mp4'),
        start_pts=float(start_pts),
        end_pts=float(end_pts),
        pts_unit='sec',
        output_format='TCHW',
    )[0]


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
    video_ids = [osp.splitext(x)[0] for x in os.listdir(video_dir)]
    video_ids = sorted(list(set(video_ids)))
    df = pd.read_csv(input_file)
    df = df[df.annotation == 1]
    df = df[df.video_id.isin(video_ids)]
    output_data = dict()

    # load model & processor
    model = DetrForObjectDetection.from_pretrained(model_name)
    model = model.to(device, dtype)
    model.eval()
    processor = DetrImageProcessor.from_pretrained(model_name)

    for indx in tqdm(range(len(df))):
        item = df.iloc[indx]
        item_id = int(item.id)
        video = read_video(
            video_dir=video_dir,
            video_id=item.video_id,
            start_pts=item.start,
            end_pts=item.end,
        )
        
        try:
            indices = sample_frame_indices(  # FIXME: hardcoded
                clip_len=8,
                frame_sample_rate=1,
                seg_len=video.shape[0],
            )
            downsampled = video[indices]
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
        except:
            pass

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()