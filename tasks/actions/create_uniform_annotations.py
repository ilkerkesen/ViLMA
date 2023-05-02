from copy import deepcopy
import json
import click
import numpy as np
from vl_bench.utils import process_path


def override(item, item1, item2, key, index1, index2):
    item.pop(key)
    item[key] = [
        item1[key][index1],
        item2[key][index2],
    ]


@click.command()
@click.option(
    '-nf', '--noun-foil-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-vf','--verb-foil-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(exists=False, file_okay=True),
    required=True,
)
@click.option(
    '--num-examples',
    type=int,
    default=512,
)
@click.option(
    '--seed',
    type=int,
    default=1,
)
@click.option(
    '--threshold',
    type=float,
    default=0.1,
)
def main(
    noun_foil_file,
    verb_foil_file,
    output_file,
    num_examples,
    seed,
    threshold,
):
    noun_foil_file = process_path(noun_foil_file)
    verb_foil_file = process_path(verb_foil_file)
    output_file = process_path(output_file)
    with open(noun_foil_file, 'r') as f:
        noun_foils = json.load(f)
    with open(verb_foil_file, 'r') as f:
        verb_foils = json.load(f)
    sampled = dict()
    num_selected = 0
    
    rng = np.random.default_rng(seed)
    keys = list(noun_foils.keys())
    rng.shuffle(keys)
    while num_selected < num_examples and len(keys) > 0:
        noun_key = keys.pop()
        verb_key = noun_key.replace('noun', 'verb')
        comb_key = noun_key.replace('foil-noun-', '')
        noun_item = noun_foils[noun_key]
        verb_item = verb_foils[verb_key]

        # select indices
        # nf_ind = rng.integers(0, len(noun_item['foils']), 1)[0]
        # vf_ind = rng.integers(0, len(verb_item['foils']), 1)[0]
        nf_ind = vf_ind = 0

        item = deepcopy(noun_item)
        for k in ('foils', 'foil_classes', 'mlm_scores', 'foiling_methods', 
                  'foiled_pos', 'foil_nli_labels', 'foil_nli_scores', 'foil_nli_pipeline'):
            override(item, noun_item, verb_item, k, nf_ind, vf_ind)
        item['proficiency'] = {
            'caption': f'there exists some {item["noun"]}.',
            'foils': [f'there exists some {item["foil_classes"][0]}.'],
        }
        sampled[comb_key] = item
        num_selected += 1
    
    with open(output_file, 'w') as f:
        json.dump(sampled, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()