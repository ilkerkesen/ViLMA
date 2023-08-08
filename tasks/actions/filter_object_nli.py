from copy import deepcopy
import json
import click
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline
from vl_bench.utils import process_path
from vl_bench.actions import (
    make_active_voice_sentence,
    make_active_voice_sentence_with_noun_phrase,
    make_foil_noun_phrases,
    make_passive_voice_sentence,
)


@click.command()
@click.option(
    '-c', '--candidate-file',
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
    '--model-name',
    type=str,
    default='ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli',
    show_default=True,
)
@click.option(
    '--device',
    type=click.Choice(choices=['cuda:0', 'cpu']),
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
def main(candidate_file, data_file, output_file, model_name, device):
    candidate_file = process_path(candidate_file)
    data_file = process_path(data_file)
    output_file = process_path(output_file)

    with open(candidate_file, 'r') as f:
        candidates = json.load(f)
    with open(data_file, 'r') as f:
        df = pd.read_csv(data_file)
        df = df[df.annotation == 1]
    filtered = dict()

    model = pipeline(
        task='text-classification',
        model=model_name,
        device=device,
        # torch_dtype=torch.float16,  # FIXME: hardcoded.
    )

    # 1. create verb/noun pairs (requires pandas tricks)
    pairs = set()
    for true_verb, true_noun in df[['verb', 'noun']].values.tolist():
        pairs.add(f'{true_verb} {true_noun}')
    pairs = sorted([tuple(x.split(' ')) for x in pairs])
    nouns = set(df.noun)

    # 2. iterate over these pairs
    for true_verb, true_noun in tqdm(pairs):
        # a. create active/passive captions and foils
        if true_verb == 'deseed':
            continue
        noun_ = true_noun
        if true_noun == "clothe":
            true_noun = "clothes"
        active_caption = make_active_voice_sentence(true_verb, true_noun)
        # passive_caption = make_passive_voice_sentence(true_verb, true_noun)

        foil_noun_phrases = candidates.get(true_verb)
        if foil_noun_phrases is None and true_verb in ('move', 'weigh'):
            excluded = nouns.difference({noun_})
            foil_noun_phrases = make_foil_noun_phrases(excluded, true_verb)
        if foil_noun_phrases is None:
            continue
        active_foils = [
            make_active_voice_sentence_with_noun_phrase(
                verb=true_verb,
                noun_phrase=item['phrase'],
            )
            for item in foil_noun_phrases
        ]
        prompts = [
            f'{active_caption}. {active_foil}.'
            for active_foil in active_foils
        ]
        model_outputs = model(prompts)
        pair = f'{true_verb} {true_noun}'
        filtered[pair] = list()
        for item, model_output in zip(foil_noun_phrases, model_outputs):
            entry = deepcopy(model_output)
            entry['noun_phrase'] = item['phrase']
            entry['noun'] = item['noun']
            entry['mlm_score'] = item['score']
            if entry['label'] != 'entailment':
                filtered[pair].append(entry)
        
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()