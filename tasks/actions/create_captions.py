import os
import os.path as osp
import json
import click
import pandas as pd
from tqdm import tqdm
from word_forms.word_forms import get_word_forms
from vl_bench.utils import process_path


def create_verb_forms_dict(file_path):
    '''
        Processes the en-verbs.txt file. Listing the most common useful forms,
        - 0: the verb lemma (e.g. throw)
        - 3: the third person singular (e.g. throws)
        - 5: the present continous tense (e.g. throwing)
        - 10: the past tense (e.g. threw)
        - 11: the past perfect tense (e.g. thrown)
    '''
    file_path = process_path(file_path)
    verb_forms_dict = dict()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if not line.startswith(';;;'):
                forms = line.strip().split(',')
                verb_forms_dict[forms[0]] = forms
    return verb_forms_dict


VERB_FORMS_FILE = osp.join(osp.dirname(process_path(__file__)), 'en-verbs.txt')
VERB_FORMS = create_verb_forms_dict(VERB_FORMS_FILE)
PRESENT_CONT_INDEX = 5
PAST_PERFECT_INDEX = 11


def get_present_continuous_tense(verb, verb_forms_dict=VERB_FORMS):
    # handle the exceptions that do not exist in the database
    if verb == 'deseed':
        return 'deseeding'

    if verb == 'microwave':
        return 'microwaving'

    forms = verb_forms_dict.get(verb)
    if forms is None:
        raise RuntimeError(f"Couldn't find the forms for {verb}")
    if forms[PRESENT_CONT_INDEX] == '':
        raise RuntimeError(
            f"The present continuous tense doesn't exist for {verb}")
    return forms[PRESENT_CONT_INDEX]


def get_past_perfect_tense(verb, verb_forms_dict=VERB_FORMS):
    # handle the exceptions that do not exist in the database
    if verb == 'deseed':
        return 'deseeded'

    if verb == 'microwave':
        return 'microwaved'

    forms = verb_forms_dict.get(verb)
    if forms is None:
        raise RuntimeError(f"Couldn't find the forms for {verb}")
    if forms[PAST_PERFECT_INDEX] == '':
        raise RuntimeError(f"The past perfect tense doesn't exist for ")
    return forms[PAST_PERFECT_INDEX]


def make_active_voice_sentence(verb, noun):
    verb_ = get_present_continuous_tense(verb)
    return f'{verb} the {noun}'


def make_passive_voice_sentence(verb, noun):
    verb_ = get_past_perfect_tense(verb)
    return f'the {noun} is being {verb_}'


@click.command()
@click.option(
    '-i', '--input',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output',
    type=click.Path(),
    required=True,
)
@click.option(
    '--active/--passive',
    default=True,
)
def main(input, output, active):
    input = osp.abspath(osp.expanduser(input))
    output = osp.abspath(osp.expanduser(output))
    passive = not active
    df = pd.read_csv(input) 
    df = df[df['annotation'] == 1]
    
    data = dict()
    for i, (dataset_idx, item) in enumerate(tqdm(df.iterrows())):
        verb, noun = item.verb.lower(), item.noun.lower()
        if active:
            caption = make_active_voice_sentence(verb, noun)
        if passive:
            caption = make_passive_voice_sentence(verb, noun)

        voice = 'active' if active else 'passive'
        key = f'{voice}-{i:04d}'

        data[key] = {
            'dataset': 'RareAct',
            'original_split': 'test',
            'dataset_idx': dataset_idx,
            'linguistic_phenomena': 'actions',
            'youtube_id': item.video_id,
            'video_file': None,
            'start_time': item.start,
            'end_time': item.end,
            'time_unit': 'sec',
            'verb': verb,
            'noun': noun,
            'voice': voice,
            'caption': caption,
        }
    
    with open(output, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
    print('done.')


if __name__ == "__main__":
    main()