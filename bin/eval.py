import json
import click
import torch
from torchmetrics.functional.classification import multiclass_accuracy, confusion_matrix, auroc
from vl_bench.utils import process_path

def format_score(val):
    return round(100*val.item(), 2)


@click.command()
@click.argument(
    'input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '--similarity/--probability',
    required=True,
)
def main(input_file, similarity):
    using_probabilities = not similarity
    input_file = process_path(input_file)
    with open(input_file, 'r') as f:
        pred = json.load(f)
    num_texts = max([len(item['scores']) for item in pred.values()])
    num_examples = len(pred)
    scores = torch.zeros(num_examples, num_texts, dtype=torch.double)
    for idx, item in enumerate(pred.values()):
        scores[idx, :len(item['scores'])] = torch.tensor(item['scores'])
    labels = torch.zeros(num_examples, dtype=torch.long)
    acc_r = multiclass_accuracy(
        scores, labels, num_classes=num_texts, average='micro')
    click.echo(f'acc_r={format_score(acc_r)}%')
    if not using_probabilities:
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