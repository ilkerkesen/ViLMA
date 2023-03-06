#!/usr/bin/env python

import os
import os.path as osp
import json

import numpy as np
import click
from tqdm import tqdm
import inflect

from vl_bench.utils import process_path


def create_example_0_index_1_diff_full_video(
    data_dir,
    entry,
    normalized=False,
    **kwargs,
):
    """
    Generates examples from full videos / full counts.
    Foil Method: -1 and +1
    Template: 0-index
    """
    method = "0-index-1-diff-full-video"
    subset = {}
    annotations_dir = 'normalized_annotations' if normalized else 'annotations'

    # read annotations
    timestamps = np.load(osp.join(
        data_dir, annotations_dir, entry['prefix'] + '.npy'))
    count = len(timestamps)

    # create the true caption from the template
    template = entry['templates'][0]
    caption = template.replace('<number>', str(count))

    start_time, end_time = 0, int(timestamps[-1])
    item_id = f'{method}-{entry["id"]}-{start_time}-{end_time}'

    foils, foiling_methods, classes_foils = [], [], []

    # -1 foil
    foils.append(template.replace('<number>', str(count-1)))
    foiling_methods.append('-1')
    classes_foils.append(count-1)

    # +1 foil
    foils.append(template.replace('<number>', str(count+1)))
    foiling_methods.append('+1')
    classes_foils.append(count+1)

    subset[item_id] = {
        'dataset': 'QUVA',
        'original_split': 'test',
        'dataset_idx': entry['id'],
        'youtube_id': None,
        'video_file': entry['prefix'] + '.mp4',
        'start_time': start_time,
        'end_time': end_time,
        'time_unit': 'pts',
        'caption': caption,
        'foils': foils,
        'foiling_methods': foiling_methods,
        'template': template,
        'classes': count,
        'classes_foils': classes_foils,
        'normalized': normalized,
    }

    return subset


def create_examples_0_index_1_diff_n_count(
    data_dir,
    entry,
    use_spelling=False,
    n_count=0,
    normalized=False,
    **kwargs,
):
    """
    Generates examples with n repetitions, specified by n_count arg.
    Foil Method: -1 and +1
    Template: 0-index
    """
    assert n_count > 0

    p = inflect.engine()
    digits_or_spelling = "use-spelling" if use_spelling else "use-digits"
    method = f"0-index-1-diff-n-count-{digits_or_spelling}"
    subset = {}
    annotations_dir = 'normalized_annotations' if normalized else 'normalized'

    # read annotations
    timestamps = np.load(osp.join(
        data_dir, annotations_dir, entry['prefix'] + '.npy'))
    total_count = len(timestamps)

    for i in range(0, total_count-n_count+1):
        assert i+n_count-1 < len(timestamps)
        count = n_count
        start_time = int(timestamps[i])
        end_time = int(timestamps[i+n_count-1])
        item_id = f'{method}-{entry["id"]}-{start_time}-{end_time}'

        # create the true caption from the template
        template = entry['templates'][0]
        caption = template.replace('<number>', str(n_count))

        foils, foiling_methods, classes_foils = [], [], []

        # -1 foil
        foils.append(template.replace('<number>', str(count-1)))
        foiling_methods.append('-1')
        classes_foils.append(n_count-1)

        # +1 foil
        foils.append(template.replace('<number>', str(count+1)))
        foiling_methods.append('+1')
        classes_foils.append(count+1)

        subset[item_id] = {
            'dataset': 'QUVA',
            'original_split': 'test',
            'dataset_idx': entry['id'],
            'youtube_id': None,
            'video_file': entry['prefix'] + '.mp4',
            'start_time': start_time,
            'end_time': end_time,
            'time_unit': 'pts',
            'caption': caption,
            'foils': foils,
            'foiling_methods': foiling_methods,
            'template': template,
            'classes': count,
            'classes_foils': classes_foils,
            'normalized': normalized,
            'digits_or_spelling': digits_or_spelling,
        }

    return subset


def create_examples_0_index_m_diff_n_count(
    data_dir,
    entry,
    use_spelling=False,
    n_count=0,
    m_diff=1,
    normalized=False,
    **kwargs,
):
    """
    Generates examples with n repetitions, specified by n_count arg.
    The foils are specified by m_diff arg (n-m and n+m).
    Template: 0-index
    """
    assert n_count > 0 and m_diff > 0

    p = inflect.engine()
    digits_or_spelling = "use-spelling" if use_spelling else "use-digits"
    method = f"0-index-{m_diff}-diff-{n_count}-count-{digits_or_spelling}"
    subset = {}
    annotations_dir = 'normalized_annotations' if normalized else 'normalized'

    # read annotations
    timestamps = np.load(osp.join(
        data_dir, annotations_dir, entry['prefix'] + '.npy'))
    timestamps = [0] + timestamps.tolist()
    total_count = len(timestamps)

    for i in range(0, total_count-n_count-2):
        assert i+n_count-m_diff < len(timestamps)
        count = n_count
        start_time = int(timestamps[i])
        end_time = int(timestamps[i+n_count])
        item_id = f'{method}-{entry["id"]}-{start_time}-{end_time}'

        # create the true caption from the template
        template = entry['templates'][0]
        gold_count = p.number_to_words(n_count) if use_spelling else n_count
        caption = template.replace('<number>', str(gold_count))

        foils, foiling_methods, classes_foils = [], [], []

        # -1 foil
        decr = max(0, n_count - m_diff)
        decr = p.number_to_words(decr) if use_spelling else str(decr)
        foils.append(template.replace('<number>', decr))
        foiling_methods.append(f'-{m_diff}')
        classes_foils.append(n_count-m_diff)

        # +1 foil
        incr = n_count + m_diff
        incr = p.number_to_words(incr) if use_spelling else str(incr)
        foils.append(template.replace('<number>', incr))
        foiling_methods.append(f'+{m_diff}')
        classes_foils.append(n_count+m_diff)

        subset[item_id] = {
            'dataset': 'QUVA',
            'original_split': 'test',
            'dataset_idx': entry['id'],
            'youtube_id': None,
            'video_file': entry['prefix'] + '.mp4',
            'start_time': start_time,
            'end_time': end_time,
            'time_unit': 'pts',
            'caption': caption,
            'foils': foils,
            'foiling_methods': foiling_methods,
            'template': template,
            'classes': count,
            'classes_foils': classes_foils,
            'normalized': normalized,
            'digits_or_spelling': digits_or_spelling,
        }

    return subset


def create_rand_margin_v1(
    data_dir,
    entry,
    use_spelling=False,
    n_count=0,
    m_diff=1,
    normalized=False,
    rng=None,
    **kwargs,
):
    """
    This function is the result of our meeting with Iacer.
        - We use all textual templates. (the final score will be computed using average)
        - We sample (uniform) two foils: smaller and larger foils.
        - C: correct count, M: margin.
        - Smaller foil: rand ~ [max(0, C-M), C-1] => FIXME: I think 0 is better.
        - Larger foil: rand ~ [C+1, C+M]
    """
    assert rng is not None
    assert m_diff > 0
    assert normalized  # We need this for some models.
    margin = m_diff

    p = inflect.engine()
    digits_or_spelling = "use-spelling" if use_spelling else "use-digits"
    method = f"rand-margin-{margin}-v1"
    subset = {}
    annotations_dir = 'normalized_annotations' if normalized else 'annotations'

    # read annotations
    timestamps = np.load(osp.join(
        data_dir, annotations_dir, entry['prefix'] + '.npy'))
    timestamps = [0] + timestamps.tolist()
    total_count = len(timestamps)

    for count in range(1, total_count):
        for start in range(0, total_count-count):
            start_time = int(timestamps[start])
            end_time = int(timestamps[start+count])
            small_f = int(rng.integers(max(0, count-margin), count))
            large_f = int(rng.integers(count+1, count+margin+1))
            small_f_t = p.number_to_words(small_f) if use_spelling else small_f
            large_f_t = p.number_to_words(large_f) if use_spelling else large_f
            count_t = p.number_to_words(count) if use_spelling else count

            templates = entry['templates']
            if count == 1:
                templates = entry['singular_templates']

            small_f_templates = large_f_templates = entry['templates']
            if small_f == 1:
                small_f_templates = entry['singular_templates']
            if large_f == 1:
                large_f_templates = entry['singular_templates']

            foiling_methods = [
                'rand ~ [max(0, C-M), C-1]',
                'rand ~ [C+1, C+M]',
            ]

            # add item for each template
            for template_id  in range(len(templates)):
                item_id = f'{method}-{entry["id"]}-{start_time}-{end_time}-{template_id}'
                template = templates[template_id]
                small_f_template = small_f_templates[template_id]
                large_f_template = large_f_templates[template_id]

                subset[item_id] = {
                    'dataset': 'QUVA',
                    'original_split': 'test',
                    'dataset_idx': entry['id'],
                    'youtube_id': None,
                    'video_file': entry['prefix'] + '.mp4',
                    'start_time': start_time,
                    'end_time': end_time,
                    'time_unit': 'pts',
                    'caption': template.replace('<number>', str(count_t)),
                    'foils': [
                        small_f_template.replace('<number>', str(small_f_t)),
                        large_f_template.replace('<number>', str(large_f_t)),
                    ],
                    'foiling_methods': foiling_methods,
                    'template': template,
                    'template_id': template_id,
                    'classes': int(count),
                    'classes_foils': [small_f, large_f],
                    'normalized': normalized,
                    'digits_or_spelling': digits_or_spelling,
                }
    return subset


METHODS = {
    '0-index-1-diff-full-video': create_example_0_index_1_diff_full_video,
    '0_index_1_diff_n_count': create_examples_0_index_1_diff_n_count,
    '0_index_m_diff_n_count': create_examples_0_index_m_diff_n_count,
    'rand_margin_v1': create_rand_margin_v1,
}


@click.command()
@click.option(
    '--input-file',
    type=click.Path(file_okay=True, exists=True, resolve_path=True),
    required=True,
)
@click.option(
    '--output-file',
    type=click.Path(writable=True, file_okay=True, resolve_path=True),
    required=True,
)
@click.option(
    '--data-dir',
    type=click.Path(dir_okay=True, exists=True, resolve_path=True),
    required=True,
)
@click.option(
    '--method',
    type=click.Choice(choices=METHODS.keys()),
    default='rand_margin_v1',
    required=True,
)
@click.option(
    '--seed',
    type=int,
    default=42,
)
@click.option(
    '--n-count',
    type=int,
    default=0,
)
@click.option(
    '--m-diff', '--margin',
    type=int,
    default=1,
)
@click.option(
    '--normalized/--unnormalized',
    default=True,
)
def main(
    input_file,
    output_file,
    data_dir,
    method,
    seed,
    n_count,
    m_diff,
    normalized,
):
    rng = np.random.default_rng(seed)
    input_file = process_path(input_file)
    output_file = process_path(output_file)
    data_dir = process_path(data_dir)

    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    data = {}
    for _, item in tqdm(raw_data.items()):
        subset = METHODS[method](
            data_dir,
            item,
            n_count=n_count,
            m_diff=m_diff,
            normalized=normalized,
            rng=rng,
        )
        data.update(subset)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    main()