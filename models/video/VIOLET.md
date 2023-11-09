# Running VIOLET
Paper Link: https://arxiv.org/abs/2111.12681

## Environment Setup
First, please setup a conda environment with python `3.9.16`.
Then, exectue the following script
```bash
pip install gdown
git clone https://github.com/andreapdr/pytorch_violet.git
cd pytorch_violet
gdown -O _snapshot/ 1B1tkA9EnlQlK72xB8liz_WRo7WTEpJDt
mkdir checkpoints
gdown -O checkpoints/ 1RLbthdRIflxCFjRTcVV5jQJGP30_lNfg
```

Install dependencies via
```bash
pip install -r requirements.txt
```

## Running VIOLET on the data

Just run the following command,
```bash
python vl_bench.py \
    --json-path /path/to/annotations.json \
    --video-dir /path/to/videos
```

This command will produce a single output `output={accuracy}%`
