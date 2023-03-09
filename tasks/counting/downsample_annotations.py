import click
import numpy as np
import json
from vl_bench.utils import process_path


@click.command()
@click.argument(
    'input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.argument(
    'output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '-N', '--num-examples',
    type=int,
    default=1000,
    required=True,
)
@click.option(
    '--seed',
    type=int,
    default=42,
)
def main(
    input_file,
    output_file,
    num_examples,
    seed,
):
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    rng = np.random.default_rng(seed)
    helper = lambda k: '-'.join(k.split('-')[:-1])
    keys = list(data.keys())
    meta_keys = sorted(list(set([helper(k) for k in keys])))
    inds = rng.permutation(len(meta_keys))[:num_examples]
    meta_keys = [meta_keys[i] for i in inds]
    keys = [k for k in keys if helper(k) in meta_keys]

    subset = dict()
    for key in keys:
        subset[key] = data[key]
    
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=4)


if __name__ == "__main__":
    main()