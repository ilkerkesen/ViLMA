import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
from transformers import AutoProcessor, AutoModel
from vl_bench.data import Dataset_v2, get_xclip_collate_fn
from vl_bench.utils import process_path


MODELS = ("microsoft/xclip-base-patch32",)


@click.command()
@click.option(
    "-i", "--input-file", type=click.Path(exists=True, file_okay=True), required=True
)
@click.option(
    "-m",
    "--model-name",
    type=click.Choice(choices=MODELS),
    default=MODELS[0],
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
)
@click.option(
    '--num-workers',
    type=int,
    default=5,
)
@click.option(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
@click.option(
    "--quva-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "--something-something-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '--youtube-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '--star-dir',
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    '-o', '--output-file',
    type=click.Path(file_okay=True),
    required=True,
)
@click.option(
    '--proficiency',
    is_flag=True,
)
@click.option("--mask-video", type=bool, required=True, default=False)
def main(
    input_file,
    model_name,
    batch_size,
    num_workers,
    device,
    quva_dir,
    something_something_dir,
    youtube_dir,
    star_dir,
    output_file,
    proficiency,
    mask_video,
):
    print(f"- running xclip on {input_file}")
    print(f"- output to {output_file}")
    # check video datasets' dirs
    assert quva_dir is not None \
        or something_something_dir is not None \
        or youtube_dir is not None
    if quva_dir is not None:
        quva_dir = process_path(quva_dir)
    if something_something_dir is not None:
        something_something_dir = process_path(something_something_dir)
    if youtube_dir is not None:
        youtube_dir = process_path(youtube_dir)
    if star_dir is not None:
        star_dir = process_path(star_dir)
    np.random.seed(0)


    # initialize model & processor
    model_name = "microsoft/xclip-base-patch32"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).half().to(device)

    # read data
    data = Dataset_v2(
        input_file,
        quva_dir=quva_dir,
        something_something_dir=something_something_dir,
        youtube_dir=youtube_dir,
        star_dir=star_dir,
        proficiency=proficiency,
    )
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_xclip_collate_fn(processor=processor),
        num_workers=num_workers,
        # pin_memory=False,
    )

    results = dict()
    for i, batch in enumerate(tqdm(loader)):
        inputs = batch['inputs'].to(device)
        num_batch_texts = batch['num_texts']
        batch_size = len(num_batch_texts)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_video.cpu()

        offset = 0
        for i in range(batch_size):
            num_texts = num_batch_texts[i]
            start = offset
            end = offset + num_texts
            scores = logits[i, start:end].flatten().tolist()
            key = batch['item_ids'][i]
            results[key] = {'scores': scores}
            offset += num_texts

    with open(process_path(output_file), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
