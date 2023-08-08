import torch
import json
import numpy as np
import click
from vl_bench.utils import process_path
from vl_bench.data import Dataset_v1
import torchvision
from torchvision.transforms import ToPILImage
import transformers
from tqdm import tqdm
import torch.nn.functional as F
from models.video.pytorch_violet.vl_bench import init_model


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
    "--star-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=False,
)
@click.option(
    "--youtube-dir",
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
    youtube_dir,
    something_something_dir,
    star_dir,
    output_file,
    mask_video,
):
    print(f"- running VIOLET on {input_file}")
    print(f"- output file: {output_file}")

    violet = init_model()
    violet.load_ckpt(
        process_path("models/video/pytorch_violet/checkpoints/ckpt_violet_pretrain.pt")
    )
    violet.to(device)
    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-uncased")

    data = Dataset_v1(
        input_file,
        youtube_dir=youtube_dir,
        something_something_dir=something_something_dir,
        star_dir=star_dir
    )

    results = {}
    with torch.no_grad():
        for item in tqdm(data):
            video = get_images(item["video"], device)
            if mask_video:
                video = torch.zeros_like(video).to(device)
            true_capt, capt_mask = str2txt(tokenizer, item["raw_texts"][0], device)
            foil_capt, foil_mask = str2txt(tokenizer, item["raw_texts"][1], device)
            true_score = get_similarity(violet, video, true_capt, capt_mask)
            foil_score = get_similarity(violet, video, foil_capt, foil_mask)
            scores = [true_score, foil_score]
            # violets output logits for each (video, cap)
            # scores = convert_to_prob(scores)
            scores = [s.to("cpu").item() for s in scores]
            results[str(item["item_id"])] = {"scores": scores}

    with open(output_file, "w") as f:
        json.dump(results, f)


def get_similarity(model, video, text, text_mask):
    score = model(video.unsqueeze(0), text.unsqueeze(0), text_mask.unsqueeze(0))
    return score


def convert_to_prob(scores):
    probs = F.softmax(torch.hstack(scores).squeeze(), dim=0).cpu()
    return [probs[0].item(), probs[1].item()]


def get_images(video, device, sample=5):
    """
    - During pre-training, we sparsely sample T = 4 video
    frames and resize them into 224x224 to split into patches
    with H = W = 32.
    - For all downstream tasks, we adopt the same video frame
    size (224x224) and patch size (32x32) but 5 sparse-sampled
    frames.

    imgs = []
    for pack in av.open(f).demux():
        for buf in pack.decode():
            if str(type(buf))=="<class 'av.video.frame.VideoFrame'>":
                imgs.append(buf.to_image().convert('RGB'))
    N = len(imgs)/(args.sample+1)

    pkl[vid] = []
    for i in range(args.sample):
        buf = io.BytesIO()
        imgs[int(N*(i+1))].save(buf, format='JPEG')
        pkl[vid].append(str(base64.b64encode(buf.getvalue()))[2:-1])

    """
    sampled_frames = []
    N = video.shape[0] / (sample + 1)

    for i in range(1, sample - 2):
        sampled_frames.append(video[int(N * (i + 1))])

    # forcing fist and last frame
    sampled_frames.insert(0, video[0])
    sampled_frames.append(video[-1])

    imgs = []
    for s in sampled_frames:
        # s = s.permute(2, 0, 1)
        s = ToPILImage()(s).convert("RGB")
        imgs.append(_str2img(s))
    img = torch.stack(imgs)

    return img.to(device)


def _str2img(img, size_img=224):
    w, h = img.size
    img = torchvision.transforms.Compose(
        [
            torchvision.transforms.Pad(
                [0, (w - h) // 2] if w > h else [(h - w) // 2, 0]
            ),
            torchvision.transforms.Resize([size_img, size_img]),
            torchvision.transforms.ToTensor(),
        ]
    )(img)
    return img


def str2txt(tokenizer, text, device, size_txt=128):
    txt = tokenizer.encode(
        text, padding="max_length", max_length=size_txt, truncation=True
    )
    mask = [1 if w != 0 else w for w in txt]
    txt, mask = np.array(txt, dtype=np.int64), np.array(mask, dtype=np.int64)
    return torch.tensor(txt).to(device), torch.tensor(mask).to(device)


if __name__ == "__main__":
    main()
