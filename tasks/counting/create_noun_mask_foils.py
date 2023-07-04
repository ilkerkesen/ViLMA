import json
import click
import torch
import spacy
from tqdm import tqdm
from transformers import pipeline
from vl_bench.utils import process_path, remove_count_phrase


TARGET_DEPS =  ('nsubjpass', 'pobj', 'dobj')
BLACKLIST = (
    'me',
    'my',
    'mine',
    'myself',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves',
    'her',
    'herself',
    'him',
    'his',
    'himself',
    'us',
    'our',
    'ourselves',
    'them',
    'their',
    'they',
    'it',
    'there',
    'and',
    'or',
    'but',
    ',',
)

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
    '--model-name',
    type=str,
    default='roberta-large',
    show_default=True,
)
@click.option(
    '--device',
    type=click.Choice(choices=['cuda:0', 'cpu']),
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '--top-k',
    type=int,
    default=16,
    show_default=True,
)
@click.option(
    '-T', '--threshold',
    type=float,
    default=0.01,
    show_default=True,
)
@click.option(
    '--include-exercise-videos',
    is_flag=True,
)
def main(
    annotation_file,
    candidate_file,
    output_file,
    model_name,
    device,
    top_k,
    threshold,
    include_exercise_videos,
):
    annotation_file = process_path(annotation_file)
    candidate_file = process_path(candidate_file)
    output_file = process_path(output_file)

    with open(annotation_file, 'r') as f:
        orig = json.load(f)

    with open(candidate_file, 'r') as f:
        candidates = json.load(f)

    nlp = spacy.load('en_core_web_lg')
    model = pipeline(
        task='fill-mask',
        model=model_name,
        device=device,
    )
    mask_token = model.tokenizer.mask_token

    # filter the templates (we already have foil candidates for some)
    templates = []
    for item_id, item in orig.items():
        if item_id in candidates:
            continue

        if not include_exercise_videos and item['category'] == 'exercise':
            continue

        for template_id, template in enumerate(item['templates']):
            templates.append((item_id, template_id, remove_count_phrase(template)))

    T = threshold
    for item_id, template_id, template in tqdm(templates):
        if item_id not in candidates:
            candidates[item_id] = dict()
        doc = nlp(template)
        inds = [i for i, t in enumerate(doc) if t.dep_ in TARGET_DEPS]
        if len(inds) == 0:
            continue
        outputs = []
        for i in inds:
            toks = [mask_token if j == i else t.text for j, t in enumerate(doc)]
            sent = ' '.join(toks).replace(' .', '.')
            this = model(sent, top_k=top_k)
            this = [x for x in this if x['score'] > T]
            this = [x for x in this if x['token_str'].strip() not in BLACKLIST]
            outputs.extend([x['sequence'] for x in this])
        candidates[item_id][template_id] = sorted(list(set(outputs)))
    
    keys = sorted([k for k, v in candidates.items() if len(v) > 0])
    filtered = {k: candidates[k] for k in keys}
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=4)


if __name__ == "__main__":
    main()