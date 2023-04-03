# Running FiT
Paper Link: https://arxiv.org/abs/2104.00650

## Environment Setup

The instructions as follows,

```bash
# clone the repository then change the directory/branch
git clone https://github.com/ilkerkesen/frozen-in-time.git
cd frozen-in-time
git checkout vl-bench origin/vl-bench

# setup the environment and directories
conda env create -f environment.yml  # I suggest micromamba.
mkdir checkpoints
mkdir data
mkdir exps

# download checkpoint and create symlinks for the dataset
wget -c https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar -P ./checkpoints/
ln -s /path/to/vl-bench/annotations ./data/vl-bench
```

## Running FiT on the data

Just run the following command,
```bash
python run_bench.py \
    --config configs/vlbench.json
    --metadata_filename json_filename.json \
    --quva_dir /path/to/quva \
    --something_something_dir /path/to/dataset-videos \
    --output_file /path/to/output.json
```

This command will produce a results annotation file `/path/to/output.json`.