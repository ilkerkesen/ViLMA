from math import prod
import json
import click
from vl_bench.utils import process_path



def calculate_score(item):
    return item['count'] + prod(item['scores'])


def ensemble_key_scores(results, key, top_k, threshold):
    candidates = dict()
    num_ensembles = len(results)
    for i in range(num_ensembles):
        this = results[i].get(key, [])
        for item in this:
            phrase = item['phrase']
            if not phrase in candidates.keys():
                candidates[phrase] = {
                    'noun': item['noun'],
                    'phrase': item['phrase'],
                    'sequence': item['sequence'],
                    'count': 1,
                    'scores': [item['score']],
                    'models': [item['model']]
                }
            else:
                candidates[phrase]['count'] += 1
                candidates[phrase]['scores'].append(item['score'])
                candidates[phrase]['models'].append(item['model'])
    for phrase in candidates.keys():
        candidates[phrase]['score'] = calculate_score(candidates[phrase])
    
    candidates = sorted(list(candidates.values()), key=lambda x: x['score'], reverse=True)
    candidates = [item for item in candidates if item['score'] >= threshold]
    candidates = candidates[:top_k]
    return candidates


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
    default=2.,
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
        candidates[key] = ensemble_key_scores(
            results=results,
            key=key,
            top_k=top_k,
            threshold=threshold,
        )

    with open(output_file, 'w') as f:
        json.dump(candidates, f, indent=4)


if __name__ == "__main__":
    main()