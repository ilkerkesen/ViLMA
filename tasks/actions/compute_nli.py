import os
import os.path as osp
import json
import click
import json
import torch
from tqdm import tqdm
from transformers import pipeline
from vl_bench.utils import process_path


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
    type=str,
    default='ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli',
    show_default=True,
)
@click.option(
    '--device',
    type=click.Choice(choices=['cuda:0', 'cpu']),
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
def main(input_file, output_file, model_name, device):
    input_file = process_path(input_file)
    output_file = process_path(output_file)

    with open(input_file, 'r') as f:
        data = json.load(f)
    new_data = dict()

    model = pipeline(
        task='text-classification',
        model=model_name,
        device=device,
        torch_dtype=torch.float16,  # FIXME: hardcoded.
    )

    for key, item in tqdm(data.items()):
        texts = [f"{item['caption']}. {foil}." for foil in item['foils']]
        outputs = model(texts)
        labels = [x['label'].lower() for x in outputs]
        scores = [x['score'] for x in outputs] 
        num_foil_candidates = len(outputs)
        item['foil_nli_labels'] = labels
        item['foil_nli_scores'] = scores
        item['foil_nli_pipeline'] = num_foil_candidates * [model_name]
        new_data[key] = item
    
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()