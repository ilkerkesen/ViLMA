import json
import click
from vl_bench.utils import process_path

@click.command()
@click.argument('inputs', nargs=-1)
@click.option('--output', required=True)
def main(inputs, output):
    merged = {}
    for input_file in inputs:
        filepath = process_path(input_file)
        with open(filepath, 'r') as f:
            merged.update(json.load(f))
    
    with open(process_path(output), 'w') as f:
        json.dump(merged, f, indent=4)


if __name__ == "__main__":
    main()