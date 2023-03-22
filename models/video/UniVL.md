# Running UniVL
Paper Link: https://arxiv.org/abs/2002.06353

## Environment Setup

First, please setup a conda environment with python 3.9.16. 
Then, clone the repository `https://github.com/andreapdr/UniVL.git`. 
Then execute the following script
```bash
git clone https://github.com/andreapdr/UniVL.git
cd UniVL
mkdir -p ./weight
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
pip install -r requirements.txt
git clone https://github.com/andreapdr/VideoFeatureExtractor.git
mkdir -p ./VideoFeatureExtractor/model
wget -P ./VideoFeatureExtractor/model https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
```

## Running UniVL on the data

Just run the following command,
```bash
python vl_bench.py \
    --json-path /path/to/annotations.json \
    --video-dir /path/to/videos
```

This command will produce a single output `output={accuracy}%`

