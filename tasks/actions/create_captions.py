import os
import os.path as osp
import json
import click
import pandas as pd
from tqdm import tqdm
from word_forms.word_forms import get_word_forms
from vl_bench.utils import process_path
from vl_bench.actions import (
    make_active_voice_sentence,
    make_passive_voice_sentence,
)


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
        json.dump(data, f, indent=4, sort_keys=False)
    print('done.')


if __name__ == "__main__":
    main()