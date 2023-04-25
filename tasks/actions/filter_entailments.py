import json
import click
from copy import deepcopy
from tqdm import tqdm
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
def main(input_file, output_file):
    input_file = process_path(input_file)
    output_file = process_path(output_file)

    with open(input_file, 'r') as f:
        data = json.load(f)
    new_data = dict()

    for key, item in tqdm(data.items()):
        num_foil_candidates = len(item['foils'])
        foils = list()
        foil_classes = list()
        mlm_scores = list()
        foiling_methods = list()
        foiled_pos = list()
        foil_nli_labels = list()
        foil_nli_scores = list()
        foil_nli_pipeline = list()
        for i in range(num_foil_candidates):
            if item['foil_nli_labels'][i] != 'entailment':
                foils.append(item['foils'][i])
                foil_classes.append(item['foil_classes'][i])
                mlm_scores.append(item['mlm_scores'][i])
                foiling_methods.append(item['foiling_methods'][i])
                foiled_pos.append(item['foiled_pos'][i])
                foil_nli_labels.append(item['foil_nli_labels'][i])
                foil_nli_scores.append(item['foil_nli_scores'][i])
                foil_nli_pipeline.append(item['foil_nli_pipeline'][i])
        new_item = deepcopy(item)
        new_item['foils'] = foils
        new_item['foil_classes'] = foil_classes
        new_item['mlm_scores'] = mlm_scores
        new_item['foiling_methods'] = foiling_methods
        new_item['foiled_pos'] = foiled_pos
        new_item['foil_nli_labels'] = foil_nli_labels
        new_item['foil_nli_scores'] = foil_nli_scores
        new_item['foil_nli_pipeline'] = foil_nli_pipeline

        new_data[key] = new_item

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()