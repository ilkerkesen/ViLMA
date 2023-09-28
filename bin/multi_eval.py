# Different from eval.py.
# Takes multiple results files to produce the combined scores.

import json
import click
import torch
# from torchmetrics.functional.classification import multiclass_accuracy, confusion_matrix, auroc
from vl_bench.utils import process_path


def format_score(val):
    return round(100*val, 2)


MODES = (
    'similarity',
    'probability',
    'perplexity',
)


@click.command()
@click.option(
    '-i', '--input-files',
    type=click.Path(exists=True, file_okay=True),
    nargs=2,
    required=True,
)
@click.option(
    '-a', '--annotation-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-m', '--mode',
    type=click.Choice(choices=MODES),
    default=MODES[0],  # similarity
    show_default=True,
)
def main(input_files, annotation_file, mode):
    input_file1, input_file2 = input_files
    input_file1 = process_path(input_file1)
    input_file2 = process_path(input_file2)
    annotation_file = process_path(annotation_file)
    with open(input_file1, 'r') as f:
        pred1 = json.load(f)
    with open(input_file2, 'r') as f:
        pred2 = json.load(f)
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    assert len(pred1) == len(pred2)
    keys = list(data.keys())

    # calculate acc_r
    num_examples = num_correct = 0
    for key in keys:
        item = data[key]
        is_main_test_valid = item['mturk']['caption'] >= 2
        is_prof_test_valid = item['proficiency']['human']['caption'] == 1
        if not (is_main_test_valid and is_prof_test_valid):
            continue
        num_examples += 1
        this_pred1 = torch.tensor(pred1[key]['scores'])
        this_pred2 = torch.tensor(pred2[key]['scores'])
        if mode != 'perplexity' and this_pred1.argmax().item() == 0 and this_pred2.argmax().item() == 0:
            num_correct += 1
        elif mode == 'perplexity' and this_pred1.argmin().item() == 0 and this_pred2.argmin().item() == 0:
            num_correct += 1
    acc_r = num_correct / num_examples
    click.echo(f'acc_r={format_score(acc_r)}%')

    if mode != 'probability':
        return


if __name__ == "__main__":
    main()