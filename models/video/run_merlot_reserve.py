import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import click
from vl_bench.data import BaseDataset
import sys
sys.path.append('../')

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp

@click.command()
@click.option(
    '--input_file',
    type=click.Path(exists=True, file_okay=True),
    required=True
)

def main(input_file):

    # read data
    data = BaseDataset(input_file)

    # initialize model & processor
    grid_size = (18, 32)
    model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

    results = dict()
    for item in tqdm(data):
        video_segments = video_to_segments(item["video_name"])
        video_segments = video_segments[item["start"]:item["stop"]]

        video_segments[0]['text'] = item["masked_sentence"]
        video_segments[0]['use_text_as_input'] = True

        for i in range(1,8):
            video_segments[i]['use_text_as_input'] = False

        video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=True)

        out_h = model.embed_video(**video_pre)
        out_h = out_h[video_pre['tokens'] == MASK]   

        options = item["foils"] 
        options += item["caption"]

        label_space = model.get_label_space(options)

        logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)

        scores = [0.0] * len(options)

        for i, logits_i in enumerate(logits):
            probs = jax.nn.softmax(logits_i, -1)
            for idx_i in jnp.argsort(-probs):
                p_i = probs[idx_i]
                scores[idx_i] = p_i * 100.0
        
        item_id = item['item_id']
        results[item_id] = scores

    n_correct = 0

    for scores in results.values():
        caption_score = scores[0]
        foil_scores = scores[1:]
        caption_predicted = False
        
        if scores[-1] == max(scores):
                caption_predicted = True
                n_correct += 1
         
    accuracy = round(100 * n_correct / len(data), 2)
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    main()