# Running VideoCLIP
Paper Link: https://aclanthology.org/2021.emnlp-main.544/

## Environment Setup

First, please setup a conda environment. You can follow the same instructions with setting up the fairseq package, but we strongly recommend using the following versions,

- python=3.8.13=ha86cf86_0_cpython
- pytorch=1.12.1=py3.8_cuda11.3_cudnn8.3.2_0
- torchaudio=0.12.1=py38_cu113
- torchvision=0.13.1=py38_cu113

```bash
git clone https://github.com/ilkerkesen/VideoCLIP.git
cd VideoCLIP
pip install -e .  # install fairseq
cd examples/MMPT
pip install -e .  # install MMPT (VideoCLIP/VLM package)
mkdir pretrained_models && cd pretrained_models
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
cd ..
python locallaunch.py projects/retri/videoclip.yaml --dryrun  # will produce error, no worries
```

## Running VideoCLIP on the data

Just run the following command,
```bash
python vl_bench.py \
    --json-path /path/to/annotations.json \
    --quva-dir /path/to/quva \
    --something-something-dir /path/to/dataset \
    --output-file /path/to/output.json
```

This command will produce a results annotation file `/path/to/output.json`.  To generate scores for the proficiency task, pass the `--proficiency` flag.