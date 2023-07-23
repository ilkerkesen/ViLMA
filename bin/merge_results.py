import click
import json
from vl_bench.utils import process_path


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
    multiple=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
def main(input_file, output_file):
    merged = dict()
    for file_path in input_file:
        file_path = process_path(file_path)
        with open(file_path, 'r') as f:
            this = json.load(f)
        merged.update(this)
    with open(process_path(output_file), 'w') as f:
        json.dump(merged, f, indent=4)


if __name__ == "__main__":
    main()