from copy import deepcopy
import json
import click
from tqdm import tqdm
import pandas as pd
import torch
from transformers import pipeline
from vl_bench.utils import process_path


ONLY_SINGULAR_NOUN_ACTIONS = (
    'shake',
    'spray',
    'unplug',
)


ONLY_PLURAL_NOUN_ACTIONS = (
    'blend',
    'deseed',
    'drink',
    'measure',
)


BLACKLIST = (
    'anything',
    'animal',
    'animals',
    'boy',
    'child',
    'children',
    'girl',
    'item',
    'kid',
    'lot',
    'object',
    'machine',
    'man',
    'nothing',
    'people',
    'person',
    'shit',
    'something',
    'someone',
    'thing',
    'things',
    'woman',

    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
)


def postprocess_outputs(outputs, T, article, model_name):
    filtered = [
        item
        for item in outputs
        if item['score'] > T and item['token_str'].strip() not in BLACKLIST
    ]

    processed = list()
    for item in filtered:
        new = {
            'noun': item['token_str'].strip(),
            'phrase': f"{article} {item['token_str'].strip()}",
            'sequence': item['sequence'],
            'score': item['score'],
            'model': model_name,
        }
        processed.append(new)
    return processed


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
    '--model-name',
    type=str,
    default='bert-base-uncased',
    show_default=True,
)
@click.option(
    '--top-k',
    type=int,
    default=256,
    show_default=True,
)
@click.option(
    '-T', '--threshold',
    type=float,
    default=0.01,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
    show_default=True,
)
@click.option(
    '--seed',
    type=int,
    default=42,
)
def main(
    input_file,
    output_file,
    model_name,
    top_k,
    threshold,
    device,
    seed,
):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    T = threshold
    df = pd.read_csv(input_file)
    df = df[df['annotation'] == 1]
    torch.random.manual_seed(seed)

    model = pipeline(task='fill-mask', model=model_name, device=device)
    mask_token = model.tokenizer.mask_token
    candidates = dict()

    verbs = sorted(list(set(df.verb)))
    prompt = 'a person can {} {} {}.'
    for verb in verbs:
        verb_ = verb
        if verb == 'drill':
            verb_ = 'drill into'
        outputs = list()
        if not verb in ONLY_PLURAL_NOUN_ACTIONS:
            # a
            text = prompt.format(verb_, 'a', mask_token)
            this = model(text, top_k=top_k)
            this = postprocess_outputs(this, T, 'a', model_name)
            outputs.extend(this)

            # an
            text = prompt.format(verb_, 'an', mask_token)
            this = model(text, top_k=top_k)
            this = postprocess_outputs(this, T, 'an', model_name)
            outputs.extend(this)
        if not verb in ONLY_SINGULAR_NOUN_ACTIONS:
            text = prompt.format(verb_, 'some', mask_token)
            this = model(text, top_k=top_k)
            this = postprocess_outputs(this, T, 'some', model_name)
            outputs.extend(this)

        candidates[verb] = outputs

    with open(output_file, 'w') as f:
        json.dump(candidates, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()