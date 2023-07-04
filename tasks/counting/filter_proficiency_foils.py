import json
import click
from tqdm import tqdm
import torch
from transformers import pipeline
from vl_bench.gruen import get_gruen_scorer
from vl_bench.utils import process_path, remove_count_phrase


@click.command()
@click.option(
    '-a', '--annotations-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-c', '--candidates-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--model-name',
    type=str,
    default='ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli',
    show_default=True,
)
@click.option(
    '-T', '--threshold',  # GRUEN threshold.
    type=float,
    default=0.8,
    show_default=True,
)
@click.option(
    '--device',
    type=str,
    default='cuda:0' if torch.cuda.is_available() else 'cpu',
)
def main(
    annotations_file,
    candidates_file,
    output_file,
    model_name,
    threshold,
    device,
):
    annotations_file = process_path(annotations_file)
    candidates_file = process_path(candidates_file)
    output_file = process_path(output_file)

    model = pipeline(
        task='text-classification',
        model=model_name,
        device=device,
    )
    compute_gruen = get_gruen_scorer()
    T = threshold

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    with open(candidates_file, 'r') as f:
        candidates = json.load(f)

    for item_id, item in annotations.items():
        item_candidates = candidates.get(item_id)
        if item_candidates is None:
            continue

        for i, template in enumerate(item['templates']):
            template_id = str(i)
            template = remove_count_phrase(template)
            template_candidates = item_candidates.get(template_id)
            if template_candidates is None:
                continue

            caption = template
            if not caption.endswith('.'):
                caption = f'{caption}.'

            gruen_scores = compute_gruen(template_candidates)
            filtered_1 = [
                text
                for (text, score) in zip(template_candidates, gruen_scores)
                if score >= T
            ]
            inputs = [f'{caption} {text}' for text in filtered_1]
            model_outputs = model(inputs)
            filtered_2 = [
                text
                for text, model_output in zip(filtered_1, model_outputs)
                if model_output['label'].lower() != 'entailment'
            ]

            candidates[item_id].pop(template_id)
            if len(filtered_2) > 0:
                candidates[item_id][template_id] = filtered_2

    keys = list(candidates.keys())
    for key in keys:
        if not candidates.get(key):
            candidates.pop(key)
    
    with open(output_file, 'w') as f:
        json.dump(candidates, f, indent=4)


if __name__ == "__main__":
    main()