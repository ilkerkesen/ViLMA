import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math

import click
import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models.video.UniVL.benchmark_univl import init_univl
from models.video.UniVL.VideoFeatureExtractor.model import init_weight
from models.video.UniVL.VideoFeatureExtractor.preprocessing import Preprocessing
from models.video.UniVL.VideoFeatureExtractor.videocnn.models import s3dg
from vl_bench.data import Dataset_v1
from vl_bench.utils import process_path


FRAMERATE_DICT = {"2d": 1, "3d": 24, "s3dg": 16, "raw_data": 16}
SIZE_DICT = {"2d": 224, "3d": 112, "s3dg": 224, "raw_data": 224}
CENTERCROP_DICT = {"2d": False, "3d": True, "s3dg": True, "raw_data": True}
SEED = 42
FEATURE_LENGTH = {"2d": 2048, "3d": 2048, "s3dg": 1024, "raw_data": 1024}

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


@click.command()
@click.option(
    "-i", "--input-file", type=click.Path(exists=True, file_okay=True), required=True
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
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
    "--coin-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "--youcook2-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "--star-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "--rareact-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(file_okay=True),
    required=True,
)
@click.option("--mask-video", type=bool, required=True, default=False)
def main(
    input_file,
    batch_size,
    device,
    quva_dir,
    something_something_dir,
    coin_dir,
    youcook2_dir,
    star_dir,
    rareact_dir,
    output_file,
    mask_video,
):
    print(f"- running UniVL on {input_file}")
    print(f"- output file: {output_file}")

    univl, tokenizer = init_univl(device=device)
    video_extractor, video_preprocessor = init_s3dg(device=device)

    # setting up cache for segmented videos
    dataset_name = input_file.split("/")[-1].split(".")[0]
    dataset_name = "change-state" if "change-state" in dataset_name else dataset_name
    CACHE_DIR = process_path(os.path.join("cache", dataset_name))
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "processed"), exist_ok=True)

    data = Dataset_v1(
        input_file,
        something_something_dir=something_something_dir,
        coin_dir=coin_dir,
        youcook2_dir=youcook2_dir,
        star_dir=star_dir,
        rareact_dir=rareact_dir,
        cache_dir=CACHE_DIR,
    )
    # TODO: there must be something wrong with the arguments and the init of UniVL ...

    results = {}
    with torch.no_grad():
        for item in tqdm(data):
            video = item["video"]
            video_path = item["video_path"]

            fname = str(item["item_id"])
            _cached_path = os.path.join(CACHE_DIR, "processed", fname + ".npy")
            if os.path.exists(_cached_path):
                video = np.load(_cached_path)
            else:
                # if end time and not cached is not none we crop videos to correct size and store them in cache
                if item["end_time"] is not None and "cache" not in item["video_path"]:
                    # rely on torchvision to cut the video (and encode webm as mp4)
                    torchvision.io.write_video(
                        # Writes a 4d tensor in [T, H, W, C] format in a video file (we get TCHW)
                        os.path.join(CACHE_DIR, fname + ".mp4"),
                        video.permute([0, 2, 3, 1]),
                        item["fps"],
                    )
                    video_path = os.path.join(CACHE_DIR, fname + ".mp4")

                # extract video features via UniVL code
                video = video_preprocessor(read_ffmpeg(video_path))
                video = extract_video_features(
                    video, video_extractor, device, batch_size=1
                )

                np.save(os.path.join(CACHE_DIR, "processed", f"{fname}.npy"), video)

            if mask_video:
                # mask video features
                video = np.zeros_like(video)

            true_capt = item["raw_texts"][0]
            foil_capt = item["raw_texts"][1]

            scores = [
                get_similarity_scores(univl, video, true_capt, tokenizer, device),
                get_similarity_scores(univl, video, foil_capt, tokenizer, device),
            ]
            # univl returns probability score for each (capt, video) pairs 
            # scores = convert_to_prob(scores)
            scores = [s.to("cpu").item() for s in scores]
            results[str(item["item_id"])] = {"scores": scores}

    with open(process_path(output_file), "w") as f:
        json.dump(results, f)


def convert_to_prob(scores):
    probs = F.softmax(torch.hstack(scores).squeeze(), dim=0).cpu()
    return [probs[0].item(), probs[1].item()]


def extract_video_features(video, video_extractor, device, batch_size=1):
    n_chunk = len(video)
    features = torch.cuda.FloatTensor(n_chunk, FEATURE_LENGTH["s3dg"]).fill_(0)
    n_iter = int(math.ceil(n_chunk / float(batch_size)))
    for i in range(n_iter):
        min_ind = i * batch_size
        max_ind = (i + 1) * batch_size
        video_batch = video[min_ind:max_ind].to(device)
        batch_features = video_extractor(video_batch)
        batch_features = F.normalize(batch_features, dim=1)
        features[min_ind:max_ind] = batch_features
    features = features.cpu().numpy()
    return features


def get_similarity_scores(univl, video_features, text, tokenizer, device):
    text_ids, text_mask, text_token_type = _get_text_input(text, tokenizer, device)
    video_mask = _get_video_mask(video_features, device)
    video_features = torch.tensor(video_features).to(device)

    capt_sequence_output, capt_visual_output = univl.get_sequence_visual_output(
        text_ids, text_mask, text_token_type, video_features, video_mask
    )
    similarity = univl.get_similarity_logits(
        capt_sequence_output, capt_visual_output, text_mask, video_mask
    )
    return similarity.item()


def _get_video_mask(video, device):
    return torch.ones(size=(1, video.shape[0])).to(device)


def _get_text_input(text, tokenizer, device):
    text_ids = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokenizer.tokenize(text.lower()) + ["[SEP]"]
    )
    text_mask = [1] * len(text_ids)
    text_token_type = [0] * len(text_ids)
    return (
        torch.tensor(text_ids).to(device),
        torch.tensor(text_mask).to(device),
        torch.tensor(text_token_type).to(device),
    )


def _get_video_dim(video_path):
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    return height, width


def _get_output_dim(h, w):
    if h >= w:
        return int(h * SIZE_DICT["s3dg"] / w), SIZE_DICT["s3dg"]
    else:
        return SIZE_DICT["s3dg"], int(w * SIZE_DICT["s3dg"] / h)


def read_ffmpeg(video_path):
    h, w = _get_video_dim(video_path)
    height, width = _get_output_dim(h, w)
    cmd = (
        ffmpeg.input(video_path)
        .filter("fps", fps=FRAMERATE_DICT["s3dg"])
        .filter("scale", width, height)
    )
    x = int((width - SIZE_DICT["s3dg"]) / 2.0)
    y = int((height - SIZE_DICT["s3dg"]) / 2.0)
    cmd = cmd.crop(x, y, SIZE_DICT["s3dg"], SIZE_DICT["s3dg"])
    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )
    height, width = SIZE_DICT["s3dg"], SIZE_DICT["s3dg"]
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    video = torch.from_numpy(video.astype("float32"))
    video = video.permute(0, 3, 1, 2)
    return video


def init_s3dg(
    device,
    model_path="models/video/UniVL/VideoFeatureExtractor/model/s3d_howto100m.pth",
):
    model = s3dg.S3D(last_fc=False)
    model = model.to(device)
    model_data = torch.load(process_path(model_path))
    model = init_weight(model, model_data)
    model.eval()

    preprocessor = Preprocessing("s3dg", FRAMERATE_DICT)

    return model, preprocessor


if __name__ == "__main__":
    main()
