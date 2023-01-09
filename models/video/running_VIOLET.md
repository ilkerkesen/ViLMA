# Running VIOLET
Paper Link: https://arxiv.org/abs/2111.12681

## Environment Setup

First, please setup a conda environment according to the original [VIOLET repo](https://github.com/tsujuifu/pytorch_violet). We recommend using the following versions,

- av=9.2.0
- tqdm=4.64.0
- opencv-python=4.6.0.66
- DALL-E=0.1
- transformers=4.19.2

```bash
git clone https://github.com/andreapdr/pytorch_violet.git
cd pytorch_violet
conda create -n violet python=3.8
```

# Data preprocessing

In order to extract the video features from the videos run the following command


```bash
# We use 4 frames during pretraining and 5 frames for downstream tasks
python _tools/my_extract_video-frame.py --path=path/to/annotations.json --outpath /path/to/output/video-features --sample=4 
```

# Running VIOLET on the data

Just run the following command,

```bash
python predict.py --json-path /path/to/annotations.json
```

This command will produce
```
TODO
```