# Running VideoCLIP
Paper Link: https://aclanthology.org/2021.emnlp-main.544/

## Environment Setup

First, please setup a conda environment,

```bash
git clone https://github.com/ilkerkesen/VideoCLIP.git
cd VideoCLIP/examples/MMPT
conda env create --file environment.yml
conda activate video-clip  # I recommend micromamba
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