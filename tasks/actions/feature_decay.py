from queue import PriorityQueue
import click
import numpy as np
from vl_bench.utils import process_path


@click.command()
@click.option(
    '-nf', '--noun-foil-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-vf','--verb-foil-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '--num-examples',
    type=int,
    default=512,
)
@click.option(
    '--seed',
    type=int,
    default=42,
)
def main(
    noun_foil_file,
    verb_foil_file,
    output_file,
):
    pass


if __name__ == "__main__":
    main()