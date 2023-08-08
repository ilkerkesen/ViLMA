from math import prod
import json
import click
from vl_bench.utils import process_path


BLACK_LIST = (
    'get',
    'have',
    'take',
    'make',
    'be',
    'carry',
    'hold',
    'afford',
    'touch',
)


@click.command()
@click.option(
    '-i', '--input-file',
    type=click.Path(exists=True, file_okay=True),
    multiple=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(exists=False, file_okay=True),
    required=True,
)
@click.option(
    '--top-k',
    type=int,
    default=5,
    show_default=True,
)
@click.option(
    '-T', '--threshold',
    type=float,
    default=0.001,
    show_default=True,
)
def main(input_file, output_file, top_k, threshold):
    input_files = [process_path(file_path) for file_path in input_file]
    input_file = None
    output_file = process_path(output_file)

    results = list()
    for input_file in input_files:
        with open(input_file, 'r') as f:
            results.append(json.load(f))
    candidates = dict()
    keys = list(results[0].keys())
    for key in keys:
        intersect = set([item['token_str'].strip() for item in results[0][key]])
        for model_results in results[1:]:
            this = set([item['token_str'].strip() for item in model_results[key]])
            intersect = intersect.intersection(this)
        scores = dict()
        for token in intersect:
            values = list()
            for res in results:
                # prior = sum([x['score'] for x in res[key]])
                prior = 1.
                val = [x['score'] for x in res[key] if x['token_str'].strip() == token][0]
                values.append(val / prior)
            # prob = sum(values) / len(results)
            prob = min(values)
            # prob = prod(values)
            if prob >= threshold:
                scores[token] = prob
        this = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        this = [x[0] for x in this[:top_k]]
        candidates[key] = this
    
    with open(output_file, 'w') as f:
        json.dump(candidates, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    main()