import json
import click
import torch
from torchmetrics.functional.classification import multiclass_accuracy, confusion_matrix, auroc
from vl_bench.utils import process_path


def format_score(val):
    return round(100*val.item(), 2)


MODES = (
    'similarity',
    'probability',
    'perplexity',
)

@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-a', '--annotation-file',
    required=True,
)
@click.option(
    '-m', '--mode',
    type=click.Choice(choices=MODES),
    required=True,
)
def main(input_file, annotation_file, mode):
    input_file = process_path(input_file)
    annotation_file = process_path(annotation_file)
    with open(input_file, 'r') as f:
        pred = json.load(f)
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    num_texts = max([len(item['scores']) for item in pred.values()])
    assert num_texts == 2

    keys = list(data.keys())
    filt_data, filt_pred = dict(), dict()
    for key in keys:
        item = data[key]
        is_main_test_valid = item['mturk']['caption'] >= 2
        is_prof_test_valid = item['proficiency']['human']['caption'] == 1
        if is_main_test_valid and is_prof_test_valid:
            filt_data[key] = item
            filt_pred[key] = pred[key]

    num_examples = len(filt_pred)
    scores = torch.zeros(num_examples, num_texts, dtype=torch.double)
    if mode == 'perplexity':
        scores.fill_(-torch.inf)
    for idx, item in enumerate(filt_pred.values()):
        item_scores = torch.tensor(item['scores'])
        item_num_texts = item_scores.numel()
        if mode == 'perplexity':
            item_scores = -item_scores
        scores[idx, :item_num_texts] = item_scores
    labels = torch.zeros(num_examples, dtype=torch.long)
    acc_r = multiclass_accuracy(
        scores, labels, num_classes=num_texts, average='micro')
    click.echo(f'acc_r={format_score(acc_r)}%')

    if mode != 'probability':
        return

    caption_probs = scores[:, 0]
    foil_probs = scores[:, 1:].flatten()
    probs = torch.cat([caption_probs, foil_probs])
    labels = torch.zeros(scores.numel(), dtype=torch.long)
    labels[:num_examples] = 1
    mat = confusion_matrix(probs, labels, task='binary')
    TP = mat[1, 1]
    FP = mat[0, 1]
    TN = mat[0, 0]
    FN = mat[1, 0]

    p_c = PPV = format_score(TP / (TP+FP))
    p_f = NPV = format_score(TN / (TN+FN))
    acc = format_score((TP + TN) / (TP + TN + FP + FN))
    auroc_val = format_score(auroc(probs, labels, task='binary'))

    print(f'p_c={p_c}%')
    print(f'p_f={p_f}%')
    print(f'min(p_c, p_f)={min(p_c, p_f)}%')
    print(f'acc={acc}%')
    print(f'auroc={auroc_val}')

if __name__ == "__main__":
    main()
