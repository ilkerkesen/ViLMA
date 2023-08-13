import numpy as np
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import click
from vl_bench.data import BaseDataset
from vl_bench.utils import process_path
import sys
sys.path.append('/.')

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

@click.option(
    '--video_dir',
    type=click.Path(exists=True),
    required=True
)

@click.option(
    '--output_dir',
    type=click.Path(exists=True),
    required=True
)

def main(input_file, video_dir, output_dir):
    # read data
    data = BaseDataset(input_file)
    output_dir = process_path(output_dir)

    # initialize model & processor
    grid_size = (18, 32)
    model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

    main_task_results = dict()
    prof_results = dict()
    error_dict = dict()
    
    for key, item in tqdm(data):
        try: 
            video_segments = video_to_segments(process_path(video_dir) + "/" + item["video_file"] + ".mp4")
            ss = item["start_time"]//5
            to = item["end_time"]//5
            
            if (to-ss) > 8:
                raise Exception("Merlot Reserve is capable of processing maximum 40 seconds length videos. Skipping..")

            video_segments = video_segments[ss:to]
            video_segments[0]['text'] = "<|MASK|>"

            video_segments[0]['use_text_as_input'] = True

            for i in range(1, to-ss): #fixme
                video_segments[i]['use_text_as_input'] = False

            video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=True)

            out_h = model.embed_video(**video_pre)
            out_h = out_h[video_pre['tokens'] == MASK]

            ##############################################################################################################

            options =  [item["caption"]]
            options += item["foils"]

            label_space = model.get_label_space(options)

            logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)

            scores = [0.0] * len(options)

            for i, logits_i in enumerate(logits):
                probs = jax.nn.softmax(logits_i, -1)
                for idx_i in jnp.argsort(-probs):
                    p_i = probs[idx_i]
                    scores[idx_i] = float(p_i * 100.0)
            
            main_task_score = scores
        
            ##############################################################################################################
            
            options =  [item["proficiency"]["caption"]]
            options += [item["proficiency"]["foiled_caption"]]  

            label_space = model.get_label_space(options)

            logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)

            scores = [0.0] * len(options)

            for i, logits_i in enumerate(logits):
                probs = jax.nn.softmax(logits_i, -1)
                for idx_i in jnp.argsort(-probs):
                    p_i = probs[idx_i]
                    scores[idx_i] = float(p_i * 100.0)
            
                    
            prof_score = scores
            
            ##############################################################################################################

            main_task_scores = dict()
            main_task_scores["scores"] = main_task_score

            prof_scores = dict()
            prof_scores["scores"] = prof_score

            main_task_results[key] = main_task_scores
            prof_results[key] = prof_scores

        except Exception as e:
            print(f"Error Occured at {key}: {e}")
            error_dict[key] = str(e)


    with open(f"{output_dir}/Main_Task_Results.json", "w") as fpm:
        json.dump(main_task_results, fpm, indent=4) 
    
    with open(f"{output_dir}/Prof_Results.json", "w") as fpp:
        json.dump(prof_results, fpp, indent=4) 
    
    with open(f"{output_dir}/Errors.json", "w") as fpe:
        json.dump(error_dict, fpe, indent=4)

if __name__ == "__main__":
    main()
