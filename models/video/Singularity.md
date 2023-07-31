# Running Singularity
Paper Link: https://arxiv.org/abs/2206.03428

## Environment Setup

We clone the repo first,

```bash
# clone the repository then change the directory/branch
git clone https://github.com/ilkerkesen/singularity.git
cd singularity
git checkout vl-bench origin/vl-bench
```

Then, we create the conda environment,
```bash
# setup the environment and directories
conda env create -f environment.yaml
conda activate sl  # I recommend micromamba
```

Download [the pretrained model checkpoints](https://nlp.cs.unc.edu/data/jielei/singularity/release/ckpts/pt.tar.gz)

Finally, setup the directories/symlinks and download the model weights,
```bash
mkdir data
cd data
wget -c https://nlp.cs.unc.edu/data/jielei/singularity/release/ckpts/pt.tar.gz
tar -xvf pt.tar.gz
rm pt.tar.gz
cd ..
```

## Running Singularity on the data

Just run the following command,
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python tasks/vl_bench.py ./configs/vl_bench \
    pretrained_path=./data/pt/singularity_temporal_17m.pth \
    ann_file=/path/to/annotation-file.json \
    quva_dir=/path/to/quva \
    something_something_dir=/path/to/dataset-videos \
    output_file=/path/to/output.json
```

This command will produce a results annotation file `/path/to/output.json`. To generate scores for the proficiency task, pass the `proficiency=True` option.