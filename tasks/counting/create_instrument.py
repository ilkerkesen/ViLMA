from copy import deepcopy
import os
import os.path as osp
import json
from tqdm import tqdm
import click
import numpy as np
from vl_bench.utils import process_path, remove_count_phrase


INSTRUMENTS = (
    'easy',
    'hard',
)


DIGITS = tuple(sorted(list(range(1, 11))))
SPELLED = (
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
)



def count_repetitions(data_dir, T=10):
    annotation_dir = osp.join(data_dir, 'normalized_annotations')
    file_names = os.listdir(annotation_dir)

    count_dict = dict()
    for file_name in file_names:
        frame_numbers = np.load(osp.join(annotation_dir, file_name)).tolist()
        num_repetitions = len(frame_numbers)
        for count in range(1, num_repetitions+1):
            if count <= T:
                num_clips = num_repetitions - count + 1
                count_dict[count] = count_dict.get(count, 0) + num_clips
    return count_dict


def make_choices(count_dict):
    choices = sorted(list(count_dict.keys()))
    counts = [count_dict[k] for k in choices]
    total = sum(counts)
    p = [c/total for c in counts]
    return choices, p


def sample_examples(
    data_dir,
    item,
    use_digits=False,
    normalized=True,
    rng=None,
    choices=None,
    p=None,
    instrument='easy',
    threshold=None,
    num_examples_per_video=None,
    max_count=10,
    proficiency_foils=None,
    **kwargs,
):
    assert rng is not None
    assert choices is not None
    assert p is not None
    assert threshold is not None
    assert normalized
    assert num_examples_per_video is not None

    T = threshold
    digits_or_spelling = 'digits' if use_digits else 'spelling'
    method = instrument
    ann_dir = osp.join(data_dir, 'normalized_annotations')
    timestamps = np.load(osp.join(ann_dir, item['prefix'] + '.npy'))
    timestamps = [0] + timestamps.tolist()
    max_count = min(max_count, len(timestamps)-1)

    possible_subclips = list()
    for t in range(len(timestamps)-1):
        for count in range(1, max_count+1):
            start = t
            end = t + count
            if end >= len(timestamps):
                break
            if instrument == 'easy' and count > T:
                continue
            if instrument == 'hard' and count <= T:
                continue
            possible_subclips.append((
                timestamps[start],
                timestamps[end],
                count,
            ))
    
    num_subclips = len(possible_subclips)
    if num_subclips >= num_examples_per_video:
        subclips = rng.choice(
            possible_subclips,
            num_examples_per_video,
            replace=False,
        ).tolist()
    else:
        subclips = deepcopy(possible_subclips)
        additional = rng.choice(
            possible_subclips,
            num_examples_per_video-len(subclips),
            replace=True,
        ).tolist()
        subclips = subclips + additional
    
    subset = dict()
    num_templates = len(item['templates'])
    for start_time, end_time, count in subclips:
        # caption & foil
        template_id = rng.integers(num_templates)
        c_template = item['templates'][template_id]
        if count == 1:
            c_template = item['singular_templates'][template_id]
        foil_count = rng.choice(choices, p=p)
        f_template = item['templates'][template_id]
        if foil_count == 1:
            f_template = item['singular_templates'][template_id]
        caption_count_str = str(count)
        foil_count_str = str(foil_count)
        if digits_or_spelling == 'spelling':
            caption_count_str = SPELLED[count-1]
            foil_count_str = SPELLED[foil_count-1]
        caption = c_template.replace('<number>', caption_count_str)
        foil = f_template.replace('<number>', foil_count_str)

        # prof. task.
        prof_tid = str(template_id)
        keys = sorted(list(proficiency_foils.keys()))
        if not prof_tid in keys:
            prof_tid = rng.choice(keys)
        prof_foil = rng.choice(proficiency_foils[prof_tid])
        prof_template = item['templates'][int(prof_tid)]
        prof_caption = remove_count_phrase(prof_template)

        item_id = '-'.join([
            'counting',
            instrument,
            str(item['id']),
            f'{start_time:04d}',
            f'{end_time:04d}',
            str(count),
            str(foil_count),
            str(template_id),
            digits_or_spelling[0],
        ])
        subset[item_id] = {
            'dataset': 'QUVA',
            'original_split': 'test',
            'dataset_idx': item['id'],
            'youtube_id': None,
            'video_file': item['prefix'] + '.mp4',
            'start_time': int(start_time),
            'end_time': int(end_time),
            'time_unit': 'pts',
            'caption': caption,
            'foils': [foil],
            'foiling_methods': ['sampling'],
            'template': item['templates'][template_id],
            'template_id': int(template_id),
            'class': int(count),
            'classes_foil': [int(foil_count)],
            'proficiency': {
                'caption': prof_caption,
                'foils': [prof_foil],
            },
            'normalized': normalized,
            'digits_or_spelling': digits_or_spelling,
        }
    return subset


@click.command()
@click.option(
    '-t', '--template-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-p', '--proficiency-file',
    type=click.Path(exists=True, file_okay=True),
    required=True,
)
@click.option(
    '-d', '--data-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--instrument',
    type=click.Choice(choices=INSTRUMENTS),
    default='easy',
    show_default=True,
)
@click.option(
    '--threshold',
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    '--max-count',
    type=int,
    default=10,
    show_default=True,
)
@click.option(
    '--seed',
    type=int,
    default=101,  # easy -> 101, hard -> 102
    show_default=True,
)
@click.option(
    '--num-examples',
    type=int,
    default=1000,
    show_default=True,
)
@click.option(
    '--use-digits',
    is_flag=True,
)
def main(
    template_file,
    proficiency_file,
    data_dir,
    output_file,
    instrument,
    threshold,
    max_count,
    seed,
    num_examples,
    use_digits,
):
    # process paths
    template_file = process_path(template_file)
    proficiency_file = process_path(proficiency_file)
    data_dir = process_path(data_dir)
    output_file = process_path(output_file)

    rng = np.random.default_rng(seed)
    with open(template_file, 'r') as f:
        task_data = json.load(f)

    with open(proficiency_file, 'r') as f:
        prof_data = json.load(f)

    # count the counts, obtain the frequencies
    count_dict = count_repetitions(data_dir, T=max_count)
    if instrument == 'easy':
        count_dict = {k: v for k, v in count_dict.items() if k > threshold}
    if instrument == 'hard':
        count_dict = {k: v for k, v in count_dict.items() if k <= threshold}
    choices, p = make_choices(count_dict)

    num_videos = len(os.listdir(osp.join(data_dir, 'normalized_videos')))
    num_examples_per_video = num_examples // num_videos
    data = dict()
    for key, item in tqdm(task_data.items()):
        examples = sample_examples(
            data_dir,
            item,
            use_digits=use_digits,
            rng=rng,
            choices=choices,
            p=p,
            instrument=instrument,
            threshold=threshold,
            num_examples_per_video=num_examples_per_video,
            max_count=max_count,
            proficiency_foils=prof_data[key],
        )
        data.update(examples)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()