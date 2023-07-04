import json
from tqdm import tqdm
import click
import spacy
import torch
from transformers import pipeline
from vl_bench.utils import process_path, remove_count_phrase
from vl_bench.actions import get_present_tense


MASK = 'can <extra_id_0>'
BLACKLIST_1 = (
    '.',
    'his',
    'her',
    'I',
    'i',
    'me',
    'myself',
    'you',
    'your',
    'yourself',
    'yourselves',
    'his',
    'him',
    'himself',
    'her',
    'herself',
    'they',
    'them',
    'themselves',
)


BLACKLIST_2 = (
    'a',
    'afford',
    'be',
    'carry',
    'do',
    'enjoy',
    'enjoy be',
    'find',
    'get',
    'grab',
    'grasp',
    'handle',
    'have',
    'help',
    'hold',
    'make',
    'own',
    'pick',
    'reach',
    'see',
    'touch',
    'take',
    'use',
    '\u2019t',
    "'t",
)

BLACKLIST_3 = (
    '.',
    '!',
    '?',
    ':',
    ';',
)

def is_valid(generated_text):
    tokens = generated_text.split()
    for token in BLACKLIST_3:
        if token in generated_text:
            return False
    if tokens[0] in BLACKLIST_2:
        return False
    for token in tokens:
        this = token.lower()
        if this in BLACKLIST_1:
            return False
    return True


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
    type=click.Choice(choices=['t5-large', 'google/flan-t5-large', 'google/flan-t5-xl']),
    default='t5-large',
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '--num-beams',
    type=int,
    default=64,
)
@click.option(
    '--do-sample',
    is_flag=True,
)
@click.option(
    '--top-p',
    type=int,
    default=None,
)
def main(
    input_file,
    output_file,
    model_name,
    device,
    num_beams,
    do_sample,
    top_p,

):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    with open(input_file, 'r') as f:
        data = json.load(f)

    nlp = spacy.load('en_core_web_lg')
    model = pipeline(
        task='text2text-generation',
        model=model_name,
        device=device,
    )

    templates = []
    for item_id, item in data.items():
        for template_id, template in enumerate(item['templates']):
            templates.append((item_id, template_id, remove_count_phrase(template)))
    
    num_verbs = 0
    candidates = dict()
    for item_id, template_id, template in tqdm(templates):
        if item_id not in candidates:
            candidates[item_id] = dict()
        doc = nlp(template)
        true_verb = None
        masked_sentence = ''
        for tok in doc:
            if tok.pos_ == 'VERB' and tok.dep_ == 'ROOT':
                true_verb = tok.text
                new_token = MASK
            else:
                new_token = tok.text
            masked_sentence = f'{masked_sentence}{new_token} '

        if true_verb is None:
            continue

        masked_sentence = f'{masked_sentence.strip().replace(" .", ".")} </s>'
        outputs = model(
            masked_sentence,
            num_beams=num_beams if not do_sample else 1,
            num_return_sequences=num_beams,
            do_sample=do_sample,
            top_p=top_p,
        )
        
        outputs = [x for x in outputs if is_valid(x['generated_text'])]
        if len(outputs) == 0:
            continue

        candidates[item_id][template_id] = list()
        for output in outputs:
            verb = output['generated_text'].split()[0]
            present_tense = get_present_tense(verb)
            predicted = output['generated_text'].replace(verb, present_tense)
            this = masked_sentence.replace(MASK, predicted).replace('. </s>', '.')
            candidates[item_id][template_id].append(this)

    keys = list(candidates.keys())
    for key in keys:
        if not candidates.get(key):
            candidates.pop(key)

    with open(output_file, 'w') as f:
        json.dump(candidates, f, indent=4)

if __name__ == "__main__":
    main()