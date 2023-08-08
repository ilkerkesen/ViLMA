import os
import os.path as osp
import json
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from vl_bench.utils import process_path
from vl_bench.actions import (
    make_active_voice_sentence,
    make_passive_voice_sentence,
    make_proficiency_sentence,
)


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-d', '--data-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--video-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    '--object-detection-file',
    type=click.Path(exists=True, file_okay=True),
)
@click.option(
    '--num-examples',
    type=int,
    default=1024,
    show_default=True,
)
@click.option(
    '--seed',
    type=int,
    default=42,
    show_default=True,
)
def main(
    input_file,
    data_file,
    output_file,
    video_dir,
    object_detection_file,
    num_examples,
    seed,
):
    input_file = process_path(input_file)
    data_file = process_path(data_file)
    output_file = process_path(output_file)
    video_dir = process_path(video_dir)
    object_detections = None
    if object_detection_file is not None:
        with open(process_path(object_detection_file), 'r') as f:
            object_detections = json.load(f)

    with open(input_file) as f:
        candidates = json.load(f)
    video_ids = [osp.splitext(x)[0] for x in os.listdir(video_dir)]
    video_ids = sorted(list(set(video_ids)))
    df = pd.read_csv(data_file)
    df = df[df.annotation == 1]
    df = df[df.video_id.isin(video_ids)]
    nouns = list(set(df.noun))

    rng = np.random.default_rng(seed)
    sampled_indices = sorted(rng.permutation(len(df))[:num_examples])

    data = dict()
    for indx in tqdm(sampled_indices):
        df_item = df.iloc[indx]
        item = {
            'dataset': 'RareAct',
            'original_split': 'test',
            'dataset_idx': int(df_item.id),
            'linguistic_phenomena': 'actions',
            'youtube_id': df_item.video_id,
            'video_file': None,
            'start_time': int(df_item.start),
            'end_time': int(df_item.end),
            'time_unit': 'sec',
        }
        true_verb, true_noun = df_item.verb, df_item.noun
        # true_noun = 'clothes' if df_item.noun == 'clothe' else true_noun
        caption = make_active_voice_sentence(true_verb, true_noun)

        # sample foil
        pair = f'{true_verb} {true_noun}'
        pair = pair + 's' if pair == 'shake clothe' else pair
        foil_candidates = candidates[pair]
        num_candidates = len(foil_candidates)
        foil_indx = rng.integers(num_candidates)
        foil_candidate = foil_candidates[foil_indx]
        foil_verb = foil_candidate['verb']
        foil = make_active_voice_sentence(foil_verb, true_noun)

        # proficiency
        detected = []
        if object_detections is not None:
            detected = object_detections.get(str(df_item.id), [])
        prof_caption = make_proficiency_sentence(true_noun)
        filtered_nouns = [x for x in nouns if x != true_noun and x not in detected]
        foil_noun = filtered_nouns[rng.integers(len(filtered_nouns))]
        prof_foil = make_proficiency_sentence(foil_noun)

        item.update({
            'caption': caption,
            'foils': [foil],
            'foil_classes': [foil_verb],
            'foil_masked_pos': ['verb'],
            'foiling_method': ['t5'],
            'foil_nli_labels': [foil_candidate['label']],
            'foil_nli_scores': [foil_candidate['score']],
            'proficiency': {
                'caption': prof_caption,
                'foils': [prof_foil],
            }
        })
        item_id = f'actions-verb-active-{indx:04d}'
        data[item_id] = item

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()