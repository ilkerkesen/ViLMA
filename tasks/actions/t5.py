from copy import deepcopy
from string import punctuation
import json
import click
from tqdm import tqdm
import pandas as pd
import torch
# from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline
from vl_bench.utils import process_path


BLACKLIST = (
    'a',
    'afford',
    'be',
    'do',
    'enjoy',
    'enjoy be',
    'find',
    'grasp',
    'have',
    'hold',
    'make',
    'own',
    'pick',
    'reach',
    'see',
    'touch',
    'take',
    'use',  # FIXME: microwave/laptop
    '\u2019t',
    "'t",
)


ADVERBS = ('easily', 'actually', 'also', 'only')


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
    default='t5-large',
    show_default=True,
)
@click.option(
    '--num-beams',
    type=int,
    default=1,
)
@click.option(
    '--num-return-sequences',
    type=int,
    default=16,
)
@click.option(
    '--max-num-sentences',
    type=int,
    default=1024,
)
@click.option(
    '--do-sample',
    is_flag=True,
)
@click.option(
    '--top-k',
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    '--top-p',
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    '-T', '--temperature',
    type=float,
    default=1.0,
    show_default=True,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
    show_default=True,
)
@click.option(
    '--masked-pos',
    type=click.Choice(choices=['noun', 'verb']),
    default='verb',
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
    num_beams,
    max_num_sentences,
    num_return_sequences,
    do_sample,
    top_k,
    top_p,
    temperature,
    device,
    masked_pos,
    seed,
):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    df = pd.read_csv(input_file)
    df = df[df['annotation'] == 1]
    torch.random.manual_seed(seed)

    model = pipeline(
        task='text2text-generation',
        model=model_name,
        device=device,
    )

    candidates = dict()
    if masked_pos == 'verb':
        nouns = sorted(list(set(df.noun)))
        prompt = 'a person can <extra_id_0> {} {}. </s>'
        for noun in tqdm(nouns):
            if noun == 'clothe':
                noun = 'clothes'

            article = 'a'
            if noun in ('apple', 'egg', 'oven'):
                article = 'an'
            if noun in ('clothes', 'eggs'):
                article = 'some'
            text = prompt.format(article, noun)

            count_dict = dict()
            num_generated = 0
            while True:
                outputs = model(
                    text,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
                for item in outputs:
                    candidate = item['generated_text'].strip()
                    if candidate[-1] in punctuation:
                        candidate = candidate[:-1]
                    tokens = candidate.split()
                    if tokens[0] in BLACKLIST:
                        continue
                    if tokens[0] in ADVERBS:
                        candidate = ' '.join(tokens[1:])
                    count_dict[candidate] = 1 + count_dict.get(candidate, 0)
                num_generated += len(outputs)
                if num_generated > max_num_sentences:
                    break
            this = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
            this = [x for x in this if x[1] >= 10]
            candidates[noun] = [x[0] for x in this]

        with open(output_file, 'w') as f:
            json.dump(candidates, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()