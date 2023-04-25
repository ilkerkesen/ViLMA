from copy import copy
import os
import os.path as osp
import json
import click
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

from vl_bench.actions import (
    make_active_voice_sentence,
    make_passive_voice_sentence,
)
from vl_bench.utils import process_path


MASK_TOKENS = {
    'roberta': '<mask>',
    'bart': '<mask>',
    'albert': '[MASK]',
}


def identify_model(model_name):
    if model_name.startswith('roberta'):
        return 'roberta'
    if model_name.startswith('albert'):
        return 'albert'
    if model_name.startswith('facebook/bart-'):
        return 'bart'
    raise NotImplementedError(f'{model_name} has not been implemented.')


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
    '--model-name',
    type=str,
    default='albert-large-v2',
    show_default=True,
)
@click.option(
    '--template',
    type=str,
    default='a person can <verb> the <noun>.',
    show_default=True,
)
@click.option(
    '--foil-verb/--foil-noun',
    default=True,
    show_default=True,
    help='foiling the verb or the noun.',
)
@click.option(
    '--top-k',
    type=int,
    default=32,
    show_default=True,
)
@click.option(
    '--device',
    type=click.Choice(choices=['cuda:0', 'cpu']),
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
def main(input_file, output_file, model_name, template, foil_verb, top_k, device):
    input_file = process_path(input_file)
    model_type = identify_model(model_name)
    mask_token = MASK_TOKENS[model_type]
    device = torch.device(device)
    dtype = torch.float16  # FIXME: hardcoded
    foil_noun = not foil_verb

    # load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    new_data = dict()

    # init the model
    model = pipeline(
        task='fill-mask',
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )

    for key, item in tqdm(data.items()):
        verb = item['verb']
        noun = item['noun']
        voice = item['voice']

        input_str = template
        if foil_verb:
            input_str = input_str.replace('<noun>', noun)
            input_str = input_str.replace('<verb>', mask_token)
        if foil_noun:
            input_str = input_str.replace('<verb>', verb)
            input_str = input_str.replace('<noun>', mask_token)

        with torch.no_grad():
            predicted = model(input_str, top_k=top_k)

        num_foil_candidates = len(predicted)  # should be equal to topk
        foil_method = f'masked language modeling using {model_name} (huggingface).'
        foiled_pos = 'verb' if foil_verb else 'noun'
        foil_classes, foils, mlm_scores = [], [], []
        for pred_item in predicted:
            foil_class = pred_item['token_str'].strip()
            if foil_class in ('either', 'through'): continue
            foil_classes.append(foil_class)
            mlm_scores.append(float(pred_item['score']))
            if voice == 'active' and foil_verb:
                foil = make_active_voice_sentence(foil_class, noun)
            elif voice == 'active' and foil_noun:
                foil = make_active_voice_sentence(verb, foil_class)
            elif voice == 'passive' and foil_verb:
                foil = make_passive_voice_sentence(foil_class, noun)
            elif voice == 'passive' and foil_noun:
                foil = make_passive_voice_sentence(verb, foil_class)
            foils.append(foil)

        item['foil_classes'] = foil_classes
        item['foils'] = foils
        item['mlm_scores'] = mlm_scores
        item['foiling_methods'] = [foil_method] * num_foil_candidates
        item['foiled_pos'] = [foiled_pos] * num_foil_candidates
        
        if foil_verb:
            new_key = f'foil-verb-{key}'
        if foil_noun:
            new_key = f'foil-noun-{key}'
        new_data[new_key] = item

    with open(process_path(output_file), 'w') as f:
        json.dump(new_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()