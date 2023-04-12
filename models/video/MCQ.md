# Running MCQ
Paper Link: https://arxiv.org/abs/2201.04850

## Environment Setup

We clone the repo first,

```bash
# clone the repository then change the directory/branch
git clone https://github.com/ilkerkesen/MCQ.git
cd MCQ
git checkout vl-bench origin/vl-bench
```

Then we create the conda environment,
```bash
# setup the environment and directories
conda create -n mcq python=3.8  # I suggest micromamba (it's way faster)
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

Finally, setup the directories/symlinks and download the model weights,
```bash
mkdir checkpoints
mkdir data
mkdir exps

# download checkpoint and create symlinks for the dataset
wget -c http://balina.ku.edu.tr/download/MCQ.pth -P ./checkpoints/
ln -s /path/to/vl-bench/annotations ./data/vl-bench
```

## Running MCQ on the data

Just run the following command,
```bash
python mcq_test_vlbench.py \
    --config configs/vlbench.json
    --metadata_filename json_filename.json \
    --quva_dir /path/to/quva \
    --something_something_dir /path/to/dataset-videos \
    --output_file /path/to/output.json
```

This command will produce a results annotation file `/path/to/output.json`.