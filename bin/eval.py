import json
import click
import torch
from torchmetrics.functional.classification import multiclass_accuracy
from vl_bench.utils import process_path


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
    using_probabilities =  not similarity
    input_file = process_path(input_file)
    with open(input_file, 'r') as f:
        pred = json.load(f)
    num_texts = max([len(item['scores']) for item in pred.values()])
    num_examples = len(pred)
    scores = torch.zeros(num_examples, num_texts, dtype=torch.double)
    for idx, item in enumerate(pred.values()):
        scores[idx, :len(item['scores'])] = torch.tensor(item['scores'])
    labels = torch.zeros(num_examples, dtype=torch.long)
    acc = multiclass_accuracy(
        scores, labels, num_classes=num_texts, average='micro')
    click.echo(f'accuracy={round(100*acc.item(), 2)}%')
    if using_probabilities:
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()