import json
import click
import numpy as np
import spacy
from tqdm import tqdm
from vl_bench.utils import process_path, remove_count_phrase


def get_gender(subj):
    toks = subj.split()
    if 'man' in toks:
        return 'male'
    if 'boy' in toks:
        return 'male'
    if 'guy' in toks:
        return 'male'

    if 'woman' in toks:
        return 'female'
    if 'girl' in toks:
        return 'female'
    
    return 'none'



@click.command()
@click.option(
    '-a', '--annotation-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-c', '--candidate-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--top-k',
    type=int,
    default=8,
    show_default=8,
)
@click.option(
    '--seed',
    type=int,
    default=42,
    show_default=True,
)
def main(
    annotation_file,
    candidate_file,
    output_file,
    top_k,
    seed,
):
    annotation_file = process_path(annotation_file)
    candidate_file = process_path(candidate_file)
    output_file = process_path(output_file)

    with open(annotation_file, 'r') as f:
        orig = json.load(f)

    with open(candidate_file, 'r') as f:
        candidates = json.load(f)
        existing_ids = sorted(list(candidates.keys()))
    rng = np.random.default_rng(seed)
    nlp = spacy.load('en_core_web_lg')

    templates = []
    for item_id, item in orig.items():
        if item['category'] != 'exercise':
            continue

        for template_id, template in enumerate(item['templates']):
            if 'same' in template:
                continue

            templates.append((
                item_id,
                template_id,
                item['action'],
                remove_count_phrase(template),
            ))

    for item_id, template_id, action, template in tqdm(templates):
        if item_id in existing_ids:
            continue

        if item_id not in candidates:
            candidates[item_id] = dict()

        foils = [x[3] for x in templates if x[2] != action]
        indx = list(rng.choice(len(foils), top_k, replace=False))
        foils = [foils[i] for i in indx]
        subj = [x for x in nlp(template).noun_chunks][0].text
        gender = get_gender(subj)

        replaced = []
        for raw in foils:
            f_subj = [x for x in nlp(raw).noun_chunks][0].text
            foil = raw.replace(f_subj, subj)
            if gender == 'male':
                foil.replace('her', 'his')
            if gender == 'female':
                foil.replace('his', 'her')
            replaced.append(foil)
        candidates[item_id][template_id] = replaced

    keys = sorted([k for k, v in candidates.items() if len(v) > 0])
    filtered = {k: candidates[k] for k in keys}
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=4)


if __name__ == "__main__":
    main()