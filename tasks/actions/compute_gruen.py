import os
import os.path as osp
import json
import click
import json
import torch
from tqdm import tqdm
from transformers import pipeline
from vl_bench.utils import process_path
from vl_bench.gruen import get_gruen_scorer


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
    '--device',
    type=click.Choice(choices=['cuda:0', 'cpu']),
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
@click.option(
    '--inject-someone',
    is_flag=True,
)
def main(input_file, output_file, device, inject_someone):
    input_file = process_path(input_file)
    output_file = process_path(output_file)

    with open(input_file, 'r') as f:
        data = json.load(f)
    new_data = dict()

    compute_gruen_scores = get_gruen_scorer()
    for key, item in tqdm(data.items()):
        texts = [item['caption']] + item['foils']
        texts = [f"{foil}." for foil in texts]
        if inject_someone:
            texts = [f"someone is {text}" for text in texts]
        gruen_scores = compute_gruen_scores(texts)
        item['caption_gruen_score'] = gruen_scores[0]
        item['foil_gruen_scores'] = gruen_scores[1:]
        new_data[key] = item

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()