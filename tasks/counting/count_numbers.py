import json
import click
import spacy
import numerizer
import pandas as pd
from tqdm import trange
from vl_bench.utils import process_path


BLACKLIST = (
    'january',      'jan',
    'feburary',     'feb',
    'march',        'mar',
    'april',        'apr',
    'may',          'may',
    'june',         'jun',
    'july',         'jul',
    'august',       'aug',
    'september',    'sep',
    'november',     'nov',
    'december',     'dec',

    'winter',
    'spring',
    'summer',
    'autumn', 'fall',
)


def in_blacklist(chunk):
    tokens = [tok.text for tok in chunk]
    for token in tokens:
        if token in BLACKLIST:
            return True
    return False


def convert_string_to_digits(text, T=50):
    try:
        result = int(text)
    except ValueError:
        return None
    else:
        return result if result <= T else None
    

def read_dataset(file_path, dataset='webvid'):
    data = list()
    if dataset == 'webvid':
        df = pd.read_csv(file_path)
        for i in range(len(df)):
            data.append(str(df.iloc[i]['name']))
    if dataset == 'vatex':
        with open(file_path, 'r') as f:
            raw = json.load(f)
        for example in raw:
            data.extend(example['enCap'])
    return data


@click.command()
@click.option(
    '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--dataset',
    type=click.Choice(choices=('webvid', 'vatex')),
    default='webvid',
)
def main(input_file, output_file, dataset):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    nlp = spacy.load('en_core_web_lg')
    data = read_dataset(input_file, dataset)
    num_captions = len(data)
    counts = dict()

    for i in trange(num_captions):
        try:
            caption = data[i]
            doc = nlp(caption)
            numerized = doc._.numerize()
            filtered = [
                convert_string_to_digits(v)
                for k, v in numerized.items()
                if not in_blacklist(k)
            ]
            filtered = [x for x in filtered if x is not None]
            for num in filtered:
                counts[num] = 1 + counts.get(num, 0)
        except:
            pass
    
    with open(output_file, 'w') as f:
        json.dump(counts, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()